package cnn

import (
	"fmt"
	"math"
	"os"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func pow2(n int) int {
	prod := 1
	for i := 0; i < n; i++ {
		prod *= 2
	}

	return prod
}

func floorToInt(x float64) int {
	return int(math.Floor(x))
}

func ceilToInt(x float64) int {
	return int(math.Ceil(x))
}

// ceil(log2 input) + 0.0001
func log2IntPlusToll(input float64) (log float64) {
	if input > 65536 || input <= 0 {
		fmt.Println("n is too large.")
		os.Exit(1)
	}
	log = -1.0

	for i := 0; i <= 16; i++ {
		pow2f := float64(pow2(i))
		if pow2f >= input {
			log = float64(i)
			if pow2f != input {
				log = log + 0.001
			}
			break
		}
	}
	return log
}

func ctZero(context *Context) *rlwe.Ciphertext {
	n := context.params_.LogMaxSlots()
	zero := make([]complex128, n)
	plain := hefloat.NewPlaintext(*context.params_, context.params_.MaxLevel())
	context.encoder_.Encode(zero, plain)
	ct, err := context.encryptor_.EncryptNew(plain)
	if err != nil {
		panic(err)
	}
	return ct
}
