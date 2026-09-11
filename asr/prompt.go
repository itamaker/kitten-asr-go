package asr

import "strings"

// buildPrompt renders kitten-asr's chat_template.jinja for a single audio-only
// turn with the given system text (almost always empty -- the template drops
// any user-supplied text content entirely, this is a pure transcription
// template, not an instruction-following one) and audioTokens copies of the
// <|audio_pad|> placeholder already expanded to match the audio encoder's
// actual output length. Hardcoded directly rather than run through a Jinja
// engine: the template's only variable behavior (system text, one audio
// block, add_generation_prompt) is fully pinned down by how this package
// always calls it.
func buildPrompt(systemText string, audioPadCount int) string {
	var b strings.Builder
	b.WriteString("<|im_start|>system\n")
	b.WriteString(systemText)
	b.WriteString("<|im_end|>\n<|im_start|>user\n<|audio_start|>")
	for i := 0; i < audioPadCount; i++ {
		b.WriteString("<|audio_pad|>")
	}
	b.WriteString("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")
	return b.String()
}
