package main

// evalItem is one synthetic (known-text, synthesized-audio) eval sample.
type evalItem struct {
	name string
	text string
}

// shortCorpus is a handful of short, varied sentences for quick WER checks
// (e.g. comparing fp32 vs int8 accuracy).
var shortCorpus = []evalItem{
	{"pangram", "The quick brown fox jumps over the lazy dog near the riverbank."},
	// Spelled with numerals, not words -- kitten-asr (like most ASR models)
	// transcribes digits as numerals, so a word-form reference would inflate
	// WER with normalization noise that has nothing to do with recognition
	// accuracy.
	{"numbers", "Flight 337 departs from gate 12 at 9:45."},
	{"question", "Could you please tell me where the nearest train station is located?"},
	{"technical", "The server restarted after applying the security patch this morning."},
	{"names", "Doctor Sarah Mitchell will present her research at the conference in Denver."},
}

// sweepParagraph is a ~30s passage, sized so a Stream window/trigger sweep
// (which re-transcribes the buffered window on every trigger, i.e. cost
// multiplies with how many triggers fire across the whole clip) stays
// tractable across a grid of settings.
const sweepParagraph = `Long distance communication has changed dramatically over the ` +
	`centuries. Ancient civilizations relied on drums and mounted messengers ` +
	`to carry information across great distances. The electric telegraph ` +
	`later allowed messages to travel across continents in minutes rather ` +
	`than weeks, and the telephone soon turned those same wires into channels ` +
	`for the human voice itself. Today a message that once took months can ` +
	`cross the planet in a fraction of a second.`

// longParagraph is a ~90-second (at espeak -s 150) multi-sentence passage,
// used to exercise Transcribe's >30s chunking and Stream's windowing/
// triggering behavior, where a single short sentence wouldn't.
const longParagraph = `The history of long distance communication begins long before the ` +
	`invention of the telegraph. Ancient civilizations relied on drums, smoke ` +
	`signals, and mounted messengers to carry information across great ` +
	`distances. In the nineteenth century, the invention of the electric ` +
	`telegraph changed everything, allowing messages to travel across ` +
	`continents in a matter of minutes rather than weeks. Samuel Morse and his ` +
	`collaborators developed a code of dots and dashes that could represent ` +
	`every letter of the alphabet, and soon telegraph lines stretched across ` +
	`Europe and North America. The telephone followed a few decades later, ` +
	`turning those same wires into channels for the human voice itself. By the ` +
	`middle of the twentieth century, radio and television had added sound and ` +
	`moving pictures to the mix, reaching audiences of millions at once. Then ` +
	`came satellites, fiber optic cables, and eventually the internet, which ` +
	`combined text, audio, and video into a single global network. Today a ` +
	`message that once took a ship several months to deliver can cross the ` +
	`planet in a fraction of a second, carried by light through glass fibers ` +
	`under the ocean floor.`
