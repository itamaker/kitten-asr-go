package main

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// synthesize renders text to a 16-bit PCM WAV via espeak-ng, caching by a
// hash of (text, speed) under cacheDir so repeated bench runs across models
// reuse the same audio instead of re-synthesizing every time.
func synthesize(cacheDir, text string, speed int) (string, error) {
	sum := sha1.Sum([]byte(fmt.Sprintf("%d|%s", speed, text)))
	path := filepath.Join(cacheDir, hex.EncodeToString(sum[:8])+".wav")
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}
	cmd := exec.Command("espeak-ng", "-w", path, "-s", fmt.Sprintf("%d", speed), text)
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("espeak-ng: %w: %s", err, out)
	}
	return path, nil
}
