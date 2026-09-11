package asr

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

// Exact port of transformers' WhisperFeatureExtractor / audio_utils.spectrogram
// and mel_filter_bank (slaney scale, slaney-normalized, NOT triangularized in
// mel space), as used by kitten-asr's preprocessor_config.json: 128 mel bins,
// 16kHz, n_fft=400, hop_length=160, always padded/truncated to a fixed 30s
// (480000-sample / 3000-mel-frame) window to match the audio encoder's own
// max_source_positions=1500 hard limit (see AudioEncoder in the exported ONNX
// graph, which only ever accepts exactly 3000 frames).
const (
	sampleRate  = 16000
	nFFT        = 400
	hopLength   = 160
	numMelBins  = 128
	chunkSecs   = 30
	numSamples  = chunkSecs * sampleRate // 480000
	numMelFrame = numSamples / hopLength // 3000
)

// melFilterBank returns the (numFreqBins, numMelBins) triangular filter
// matrix used to project a power spectrogram onto mel bins.
func melFilterBank() [][]float64 {
	const (
		numFreqBins = nFFT/2 + 1 // 201
		minFreq     = 0.0
		maxFreq     = 8000.0
	)
	melMin := hertzToMelSlaney(minFreq)
	melMax := hertzToMelSlaney(maxFreq)

	filterFreqs := make([]float64, numMelBins+2)
	for i := range filterFreqs {
		mel := melMin + (melMax-melMin)*float64(i)/float64(numMelBins+1)
		filterFreqs[i] = melToHertzSlaney(mel)
	}

	fftFreqs := make([]float64, numFreqBins)
	for i := range fftFreqs {
		fftFreqs[i] = float64(sampleRate/2) * float64(i) / float64(numFreqBins-1)
	}

	filters := make([][]float64, numFreqBins)
	for i := range filters {
		filters[i] = make([]float64, numMelBins)
	}
	for k := 0; k < numMelBins; k++ {
		filterDiffDown := filterFreqs[k+1] - filterFreqs[k]
		filterDiffUp := filterFreqs[k+2] - filterFreqs[k+1]
		enorm := 2.0 / (filterFreqs[k+2] - filterFreqs[k])
		for i := 0; i < numFreqBins; i++ {
			downSlope := -(filterFreqs[k] - fftFreqs[i]) / filterDiffDown
			upSlope := (filterFreqs[k+2] - fftFreqs[i]) / filterDiffUp
			v := math.Min(downSlope, upSlope)
			if v < 0 {
				v = 0
			}
			filters[i][k] = v * enorm
		}
	}
	return filters
}

func hertzToMelSlaney(freq float64) float64 {
	const minLogHertz = 1000.0
	const minLogMel = 15.0
	logstep := 27.0 / math.Log(6.4)
	if freq >= minLogHertz {
		return minLogMel + math.Log(freq/minLogHertz)*logstep
	}
	return 3.0 * freq / 200.0
}

func melToHertzSlaney(mel float64) float64 {
	const minLogHertz = 1000.0
	const minLogMel = 15.0
	logstep := math.Log(6.4) / 27.0
	if mel >= minLogMel {
		return minLogHertz * math.Exp(logstep*(mel-minLogMel))
	}
	return 200.0 * mel / 3.0
}

// hannWindow returns the *periodic* Hann window (transformers' window_function
// default, periodic=True): equivalent to a symmetric (n+1)-point Hann window
// with its last sample dropped, i.e. denominator n rather than n-1.
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	}
	return w
}

// featureExtractor computes log-mel spectrograms matching WhisperFeatureExtractor
// exactly, always producing a fixed (numMelBins, numMelFrame) = (128, 3000)
// output (silence-padded if the audio is shorter than 30s, truncated if longer
// -- matching kitten-asr's audio encoder, which only accepts this fixed size).
type featureExtractor struct {
	window  []float64
	filters [][]float64 // (numFreqBins, numMelBins)
	fft     *fourier.FFT
}

func newFeatureExtractor() *featureExtractor {
	return &featureExtractor{
		window:  hannWindow(nFFT),
		filters: melFilterBank(),
		fft:     fourier.NewFFT(nFFT),
	}
}

// LogMelSpectrogram takes mono float32 PCM samples at 16kHz and returns a
// flattened (numMelBins*numMelFrame) row-major [mel][frame] array, plus the
// number of frames that came from real (non-padding) audio.
func (fe *featureExtractor) LogMelSpectrogram(samples []float32) (mel []float32, realFrames int) {
	wave := make([]float64, numSamples)
	n := len(samples)
	if n > numSamples {
		n = numSamples
	}
	for i := 0; i < n; i++ {
		wave[i] = float64(samples[i])
	}
	realFrames = n / hopLength
	if n%hopLength != 0 {
		realFrames++
	}
	if realFrames > numMelFrame {
		realFrames = numMelFrame
	}

	// Center-pad by nFFT/2 on each side with reflection, matching
	// numpy.pad(..., mode="reflect").
	padded := reflectPad(wave, nFFT/2)

	numFrames := 1 + (len(padded)-nFFT)/hopLength // = numMelFrame+1; the extractor drops the last frame
	frameMel := make([][]float64, numFrames)

	buf := make([]float64, nFFT)
	coeff := make([]complex128, nFFT/2+1)
	power := make([]float64, nFFT/2+1)
	for f := 0; f < numFrames; f++ {
		start := f * hopLength
		for i := 0; i < nFFT; i++ {
			buf[i] = padded[start+i] * fe.window[i]
		}
		fe.fft.Coefficients(coeff, buf)
		for i, c := range coeff {
			power[i] = real(c)*real(c) + imag(c)*imag(c)
		}
		row := make([]float64, numMelBins)
		for k := 0; k < numMelBins; k++ {
			var sum float64
			for i := 0; i < nFFT/2+1; i++ {
				sum += fe.filters[i][k] * power[i]
			}
			if sum < 1e-10 {
				sum = 1e-10
			}
			row[k] = math.Log10(sum)
		}
		frameMel[f] = row
	}
	frameMel = frameMel[:numMelFrame] // drop the extra trailing frame, matching log_spec[:, :-1]

	maxVal := math.Inf(-1)
	for _, row := range frameMel {
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
	}
	floor := maxVal - 8.0

	mel = make([]float32, numMelBins*numMelFrame)
	for f, row := range frameMel {
		for k, v := range row {
			if v < floor {
				v = floor
			}
			v = (v + 4.0) / 4.0
			mel[k*numMelFrame+f] = float32(v) // [mel][frame] row-major, matching input_features (1, 128, 3000)
		}
	}
	return mel, realFrames
}

// reflectPad pads x with pad samples of numpy-style reflection on each side:
// [..., x[2], x[1]] + x + [x[n-2], x[n-3], ...] (edge sample itself is not
// repeated, matching numpy.pad(mode="reflect") default).
func reflectPad(x []float64, pad int) []float64 {
	n := len(x)
	out := make([]float64, n+2*pad)
	for i := 0; i < pad; i++ {
		out[pad-1-i] = x[(i+1)%n]
		out[pad+n+i] = x[n-2-i]
	}
	copy(out[pad:pad+n], x)
	return out
}
