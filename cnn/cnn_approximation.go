package cnn

import (
	"github.com/lifejade/cnn_ckksv5/cnn/comp"
	"github.com/lifejade/cnn_ckksv5/cnn/scaleinv"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func EvalApproxMinimaxReLU(cipherin *rlwe.Ciphertext, alpha int, context Context) (destination *rlwe.Ciphertext) {

	if alpha == 13 {

		compNo := 3
		deg := []int{15, 15, 27}
		scaledVal := 1.6
		var tree []comp.Tree

		for i := 0; i < compNo; i++ {

			tr := comp.OddBaby(deg[i])
			tree = append(tree, *tr)
			// tr.Print()
		}

		scaleContext := scaleinv.ScaleContext{
			Encoder_:   context.encoder_,
			Encryptor_: context.encryptor_,
			Decryptor_: context.decryptor_,
			Eval_:      context.eval_,
			Params_:    *context.params_,
		}

		destination = comp.MinimaxReLU(compNo, alpha, deg, tree, scaledVal, scaleContext, cipherin)

	}

	return destination
}
