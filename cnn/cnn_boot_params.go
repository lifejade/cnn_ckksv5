package cnn

import (
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
)

type defaultParametersLiteral struct {
	SchemeParams        hefloat.ParametersLiteral
	BootstrappingParams bootstrapping.ParametersLiteral
}

var Test_Parameters = defaultParametersLiteral{
	hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{60, 45, 45, 45, 45, 45},
		LogP:            []int{61, 61, 61, 61},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 45,
	},
	bootstrapping.ParametersLiteral{
		SlotsToCoeffsFactorizationDepthAndLogScales: [][]int{{42}, {42}, {42}},
		CoeffsToSlotsFactorizationDepthAndLogScales: [][]int{{58}, {58}, {58}, {58}},
		LogMessageRatio: utils.Pointy(2),
		Mod1InvDegree:   utils.Pointy(7),
	},
}

var CNN_Cifar10_Parameters = defaultParametersLiteral{
	SchemeParams: hefloat.ParametersLiteral{
		// CIFAR-10
		// index [0]
		// logN = 16, full slots
		// logq = 51, logp = 46
		// scale = 1<<46
		// # special modulus = 3
		// # available levels = 16
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51, 51, 51},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 46,
	},
	BootstrappingParams: bootstrapping.ParametersLiteral{
		SlotsToCoeffsFactorizationDepthAndLogScales: [][]int{{51}, {51}, {51}},
		CoeffsToSlotsFactorizationDepthAndLogScales: [][]int{{51}, {51}, {51}},
		LogMessageRatio: utils.Pointy(5),
		DoubleAngle:     utils.Pointy(2),
		Mod1Degree:      utils.Pointy(63),
		K:               utils.Pointy(25),
		EvalModLogScale: utils.Pointy(51),
	},
}
