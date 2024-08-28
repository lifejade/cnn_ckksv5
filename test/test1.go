package main

import (
	"fmt"

	"github.com/lifejade/cnn_ckksv5/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
)

// boot params test
func main() {

	ckksParams := cnn.CNN_Cifar10_Parameters

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, ckksParams.BootstrappingParams)
	if err != nil {
		panic(err)
	}

	// generate classes
	kgen := rlwe.NewKeyGenerator(params)

	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)

	fmt.Println("make boot params")
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	btp, _ := bootstrapping.NewEvaluator(btpParams, btpevk)
	_, _, _, _, _ = encryptor, decryptor, encoder, evaluator, btp

	fmt.Println("end boot params")

	n := 1 << params.LogMaxSlots()
	value := make([]complex128, n)
	for i := range value {
		//value[i] = sampling.RandComplex128(-1, 1)
		value[i] = complex(0.05, 0)
		//value[i] = complex(float64(i%30), 0)
	}

	plaintext := hefloat.NewPlaintext(params, 16)
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	DecryptPrint(params, cipher, *decryptor, *encoder)

	res, _ := btp.Bootstrap(cipher)

	DecryptPrint(params, res, *decryptor, *encoder)
}

func DecryptPrint(params hefloat.Parameters, ciphertext *rlwe.Ciphertext, decryptor rlwe.Decryptor, encoder hefloat.Encoder) {

	N := 1 << params.LogN()
	n := N / 2
	message := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(ciphertext), message)

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", ciphertext.LogScale())
	fmt.Printf("Values: %6.10f %6.10f %6.10f %6.10f %6.10f...\n", message[0], message[1], message[2], message[3], message[4])

	// max, min
	max, min := 0.0, 1.0
	for _, v := range message {
		if max < real(v) {
			max = real(v)
		}
		if min > real(v) {
			min = real(v)
		}
	}

	fmt.Println("Max, Min value: ", max, " ", min)
	fmt.Println()

	return

}
