package cnn

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func CompactGappedConvolution(cnnIn TensorCipher, co, st, fh, fw int, data, runningVar, constantWeight []float64, epsilon float64, context *Context) (cnnOut TensorCipher) {
	// set parameters
	ki, hi, wi, ci, ti, pi, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.p_, cnnIn.logn_
	var ko, ho, wo, to, po int

	// evaluator
	eval := *context.eval_

	_, _, _ = po, ti, eval
	// error check
	if st != 1 && st != 2 {
		fmt.Println("supported st is only 1 or 2")
		os.Exit(1)
	}
	if len(data) != fh*fw*ci*co {
		fmt.Println("the size of data vector is not ker x ker x h x h")
		os.Exit(1)
	}
	if len(runningVar) != co || len(constantWeight) != co {
		fmt.Println("the size of running_var or weight is not correct")
		os.Exit(1)
	}

	if st == 1 {
		ho, wo, ko = hi, wi, ki
	} else if st == 2 {
		ho, wo, ko = hi/2, wi/2, ki*2
	}
	log2Int := func(n int) int {
		if n > 65536 || n <= 0 {
			fmt.Println("n is too large.")
			os.Exit(1)
		}
		d := -1
		for i := 0; i <= 16; i++ {
			if pow2(i) == n {
				d = i
				break
			}
		}
		return d
	}

	n := 1 << logn
	to = (co + ko*ko - 1) / (ko * ko)
	po = pow2(floorToInt(math.Log(float64(n)/float64(ko*ko*ho*wo*to)) / math.Log(2)))
	q := (co + pi - 1) / pi

	// check if pi, po | n
	if n%pi != 0 {
		fmt.Println("n is not divisible by pi")
		os.Exit(1)
	}
	if n%po != 0 {
		fmt.Println("n is not divisible by po")
		os.Exit(1)
	}
	startTime := time.Now()

	// weight setting
	weight := make([][][][]float64, fh)
	for i1 := 0; i1 < fh; i1++ {
		weight[i1] = make([][][]float64, fw)
		for i2 := 0; i2 < fw; i2++ {
			weight[i1][i2] = make([][]float64, ci)
			for j3 := 0; j3 < ci; j3++ {
				weight[i1][i2][j3] = make([]float64, co)
				for j4 := 0; j4 < co; j4++ {
					weight[i1][i2][j3][j4] = data[fh*fw*ci*j4+fh*fw*j3+fw*i1+i2]
				}
			}
		}
	}

	// compact shifted weight vector setting
	compactWeightVec := make([][][]*rlwe.Plaintext, fh)
	for i1 := 0; i1 < fh; i1++ {
		compactWeightVec[i1] = make([][]*rlwe.Plaintext, fw)
		for i2 := 0; i2 < fw; i2++ {
			compactWeightVec[i1][i2] = make([]*rlwe.Plaintext, q)
			for i9 := 0; i9 < q; i9++ {
				compactWeightVec[i1][i2][i9] = hefloat.NewPlaintext(*context.params_, 16)
				arr := make([]complex128, n)
				for j8 := 0; j8 < n; j8++ {
					j5, j6, i7, i8 := ((j8%(n/pi))%(ki*ki*hi*wi))/(ki*wi), (j8%(n/pi))%(ki*wi), (j8%(n/pi))/(ki*ki*hi*wi), j8/(n/pi)
					if j8%(n/pi) >= ki*ki*hi*wi*ti || i8+pi*i9 >= co || ki*ki*i7+ki*(j5%ki)+j6%ki >= ci || (j6/ki)-(fw-1)/2+i2 < 0 || (j6/ki)-(fw-1)/2+i2 > wi-1 || (j5/ki)-(fh-1)/2+i1 < 0 || (j5/ki)-(fh-1)/2+i1 > hi-1 {
						arr[j8] = 0.0
					} else {
						arr[j8] = complex(weight[i1][i2][ki*ki*i7+ki*(j5%ki)+j6%ki][i8+pi*i9], 0)
					}
				}
				context.encoder_.Encode(arr, compactWeightVec[i1][i2][i9])
			}
		}
	}

	// select one setting
	selectOne := make([][][][]float64, co)
	for j4 := 0; j4 < co; j4++ {
		selectOne[j4] = make([][][]float64, ko*ho)
		for v1 := 0; v1 < ko*ho; v1++ {
			selectOne[j4][v1] = make([][]float64, ko*wo)
			for v2 := 0; v2 < ko*wo; v2++ {
				selectOne[j4][v1][v2] = make([]float64, to)
				for u3 := 0; u3 < to; u3++ {
					if ko*ko*u3+ko*(v1%ko)+v2%ko == j4 {
						selectOne[j4][v1][v2][u3] = constantWeight[j4] / math.Sqrt(runningVar[j4]+epsilon)
					} else {
						selectOne[j4][v1][v2][u3] = 0.0
					}
				}
			}
		}
	}

	// select one vector setting
	selectOneVec := make([][]float64, co)
	for j4 := 0; j4 < co; j4++ {
		selectOneVec[j4] = make([]float64, 1<<logn)
	}
	for j4 := 0; j4 < co; j4++ {
		for v1 := 0; v1 < ko*ho; v1++ {
			for v2 := 0; v2 < ko*wo; v2++ {
				for u3 := 0; u3 < to; u3++ {
					selectOneVec[j4][ko*ko*ho*wo*u3+ko*wo*v1+v2] = selectOne[j4][v1][v2][u3]
				}
			}
		}
	}
	elapse := time.Since(startTime)
	fmt.Printf("weight & selectone : %s\n", elapse)
	startTime = time.Now()

	cnnCtxIn := cnnIn.cipher_
	initScale := cnnCtxIn.Scale

	if fh%2 == 0 || fw%2 == 0 {
		fmt.Println("fh and fw should be odd")
		os.Exit(1)
	}
	// rotated input precomputation
	ctxtRot := make([][]*rlwe.Ciphertext, fh)
	for i1 := 0; i1 < fh; i1++ {
		ctxtRot[i1] = make([]*rlwe.Ciphertext, fw)
		for i2 := 0; i2 < fw; i2++ {
			ctxtRot[i1][i2] = cnnCtxIn.CopyNew()
			eval.Rotate(ctxtRot[i1][i2], ki*ki*wi*(i1-(fh-1)/2)+ki*(i2-(fw-1)/2), ctxtRot[i1][i2])
		}
	}
	elapse = time.Since(startTime)
	fmt.Printf("rotated input precomputation : %s\n", elapse)
	startTime = time.Now()
	ctZero := ctZero(context)
	ctd := ctZero.CopyNew()
	_, _ = ctZero, initScale

	for i3 := 0; i3 < q; i3++ {
		ctb := ctZero.CopyNew()
		for i1 := 0; i1 < fh; i1++ {
			for i2 := 0; i2 < fw; i2++ {
				temp, _ := eval.MulNew(ctxtRot[i1][i2], compactWeightVec[i1][i2][i3])
				eval.Add(ctb, temp, ctb)
			}
		}

		eval.RescaleTo(ctb, initScale, ctb)

		inter := ctb.CopyNew()
		var sum *rlwe.Ciphertext
		var temp *rlwe.Ciphertext
		// summation for all input channels
		d, c := log2Int(ki), log2Int(ti)
		for x := 0; x < d; x++ {
			temp = inter.CopyNew()
			eval.Rotate(temp, pow2(x), temp)
			// RotatePrintCIFAR10Convolution(temp, pow2(x), temp, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
			eval.Add(inter, temp, inter)
		}
		for x := 0; x < d; x++ {
			temp = inter.CopyNew()
			eval.Rotate(temp, pow2(x)*ki*wi, temp)
			// RotatePrintCIFAR10Convolution(temp, pow2(x)*ki*wi, temp, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
			eval.Add(inter, temp, inter)
		}
		if c == -1 {
			sum = ctZero.CopyNew()
			for x := 0; x < ti; x++ {
				temp = inter.CopyNew()
				eval.Rotate(temp, ki*ki*hi*wi*x, temp)
				// RotatePrintCIFAR10Convolution(temp, ki*ki*hi*wi*x, temp, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
				eval.Add(sum, temp, sum)
			}
			inter = sum.CopyNew()
		} else {
			for x := 0; x < c; x++ {
				temp = inter.CopyNew()
				eval.Rotate(inter, pow2(x)*ki*ki*hi*wi, inter)
				// RotatePrintCIFAR10Convolution(inter, pow2(x)*ki*ki*hi*wi, inter, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
				eval.Add(inter, temp, inter)
			}
		}
		ctc := inter.CopyNew()

		for i4 := 0; (i4 < pi) && (i4+pi*i3 < co); i4++ {
			i := pi*i3 + i4
			temp, _ := eval.RotateNew(ctc, -(i/(ko*ko))*ko*ko*ho*wo+(n/pi)*(i%pi)-((i%(ko*ko))/ko)*ko*wo-(i%ko))
			eval.Mul(temp, selectOneVec[i], temp)
			eval.Add(ctd, temp, ctd)
		}
	}
	eval.RescaleTo(ctd, initScale, ctd)
	elapse = time.Since(startTime)
	fmt.Printf("etc : %s\n", elapse)
	startTime = time.Now()

	result := ctd.CopyNew()
	for j := 0; j < int(log2IntPlusToll(float64(po))); j++ {
		temp, _ := eval.RotateNew(result, -pow2(j)*(n/po))
		eval.Add(result, temp, result)
	}
	elapse = time.Since(startTime)
	fmt.Printf("rotate : %s\n", elapse)

	cnnOut = TensorCipher{
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      po,
		logn_:   logn,
		cipher_: result,
	}
	return cnnOut
}

func CompactGappedConvolution2(cnnIn TensorCipher, co, st, fh, fw int, data, runningVar, constantWeight []float64, epsilon float64, context *Context) (cnnOut TensorCipher) {
	// set parameters
	ki, hi, wi, ci, ti, pi, logn := cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.t_, cnnIn.p_, cnnIn.logn_
	var ko, ho, wo, to, po int

	// evaluator
	eval := context.eval_

	_, _, _ = po, ti, eval
	// error check
	if st != 1 && st != 2 {
		fmt.Println("supported st is only 1 or 2")
		os.Exit(1)
	}
	if len(data) != fh*fw*ci*co {
		fmt.Println("the size of data vector is not ker x ker x h x h")
		os.Exit(1)
	}
	if len(runningVar) != co || len(constantWeight) != co {
		fmt.Println("the size of running_var or weight is not correct")
		os.Exit(1)
	}

	if st == 1 {
		ho, wo, ko = hi, wi, ki
	} else if st == 2 {
		ho, wo, ko = hi/2, wi/2, ki*2
	}
	log2Int := func(n int) int {
		if n > 65536 || n <= 0 {
			fmt.Println("n is too large.")
			os.Exit(1)
		}
		d := -1
		for i := 0; i <= 16; i++ {
			if pow2(i) == n {
				d = i
				break
			}
		}
		return d
	}

	n := 1 << logn
	to = (co + ko*ko - 1) / (ko * ko)
	po = pow2(floorToInt(math.Log(float64(n)/float64(ko*ko*ho*wo*to)) / math.Log(2)))
	q := (co + pi - 1) / pi

	// check if pi, po | n
	if n%pi != 0 {
		fmt.Println("n is not divisible by pi")
		os.Exit(1)
	}
	if n%po != 0 {
		fmt.Println("n is not divisible by po")
		os.Exit(1)
	}
	starttime := time.Now()
	// weight setting
	weight := make([][][][]float64, fh)
	for i1 := 0; i1 < fh; i1++ {
		weight[i1] = make([][][]float64, fw)
		for i2 := 0; i2 < fw; i2++ {
			weight[i1][i2] = make([][]float64, ci)
			for j3 := 0; j3 < ci; j3++ {
				weight[i1][i2][j3] = make([]float64, co)
				for j4 := 0; j4 < co; j4++ {
					weight[i1][i2][j3][j4] = data[fh*fw*ci*j4+fh*fw*j3+fw*i1+i2]
				}
			}
		}
	}

	// compact shifted weight vector setting
	compactWeightVec := make([][][][]float64, fh)
	for i1 := 0; i1 < fh; i1++ {
		compactWeightVec[i1] = make([][][]float64, fw)
		for i2 := 0; i2 < fw; i2++ {
			compactWeightVec[i1][i2] = make([][]float64, q)
			for i9 := 0; i9 < q; i9++ {
				compactWeightVec[i1][i2][i9] = make([]float64, n)
				for j8 := 0; j8 < n; j8++ {
					j5, j6, i7, i8 := ((j8%(n/pi))%(ki*ki*hi*wi))/(ki*wi), (j8%(n/pi))%(ki*wi), (j8%(n/pi))/(ki*ki*hi*wi), j8/(n/pi)
					if j8%(n/pi) >= ki*ki*hi*wi*ti || i8+pi*i9 >= co || ki*ki*i7+ki*(j5%ki)+j6%ki >= ci || (j6/ki)-(fw-1)/2+i2 < 0 || (j6/ki)-(fw-1)/2+i2 > wi-1 || (j5/ki)-(fh-1)/2+i1 < 0 || (j5/ki)-(fh-1)/2+i1 > hi-1 {
						compactWeightVec[i1][i2][i9][j8] = 0.0
					} else {
						compactWeightVec[i1][i2][i9][j8] = weight[i1][i2][ki*ki*i7+ki*(j5%ki)+j6%ki][i8+pi*i9]
					}
				}
			}
		}
	}

	// select one setting
	selectOne := make([][][][]float64, co)
	for j4 := 0; j4 < co; j4++ {
		selectOne[j4] = make([][][]float64, ko*ho)
		for v1 := 0; v1 < ko*ho; v1++ {
			selectOne[j4][v1] = make([][]float64, ko*wo)
			for v2 := 0; v2 < ko*wo; v2++ {
				selectOne[j4][v1][v2] = make([]float64, to)
				for u3 := 0; u3 < to; u3++ {
					if ko*ko*u3+ko*(v1%ko)+v2%ko == j4 {
						selectOne[j4][v1][v2][u3] = constantWeight[j4] / math.Sqrt(runningVar[j4]+epsilon)
					} else {
						selectOne[j4][v1][v2][u3] = 0.0
					}
				}
			}
		}
	}

	// select one vector setting
	selectOneVec := make([][]float64, co)
	for j4 := 0; j4 < co; j4++ {
		selectOneVec[j4] = make([]float64, 1<<logn)
	}
	for j4 := 0; j4 < co; j4++ {
		for v1 := 0; v1 < ko*ho; v1++ {
			for v2 := 0; v2 < ko*wo; v2++ {
				for u3 := 0; u3 < to; u3++ {
					selectOneVec[j4][ko*ko*ho*wo*u3+ko*wo*v1+v2] = selectOne[j4][v1][v2][u3]
				}
			}
		}
	}

	cnnCtxIn := cnnIn.cipher_
	initScale := cnnCtxIn.Scale

	elapse := time.Since(starttime)
	fmt.Println("init params : ", elapse)
	starttime = time.Now()

	if fh%2 == 0 || fw%2 == 0 {
		fmt.Println("fh and fw should be odd")
		os.Exit(1)
	}
	// rotated input precomputation
	ctxtRot := make([][]*rlwe.Ciphertext, fh)
	for i1 := 0; i1 < fh; i1++ {
		ctxtRot[i1] = make([]*rlwe.Ciphertext, fw)
		for i2 := 0; i2 < fw; i2++ {
			ctxtRot[i1][i2] = cnnCtxIn.CopyNew()
			eval.Rotate(ctxtRot[i1][i2], ki*ki*wi*(i1-(fh-1)/2)+ki*(i2-(fw-1)/2), ctxtRot[i1][i2])
		}
	}
	ctZero := ctZero(context)
	ctd := ctZero.CopyNew()
	_, _ = ctZero, initScale

	elapse = time.Since(starttime)
	fmt.Println("rotated input precomputation : ", elapse)
	starttime = time.Now()

	for i3 := 0; i3 < q; i3++ {
		stt := time.Now()
		ctb := ctZero.CopyNew()
		for i1 := 0; i1 < fh; i1++ {
			for i2 := 0; i2 < fw; i2++ {
				temp, _ := eval.MulNew(ctxtRot[i1][i2], compactWeightVec[i1][i2][i3])
				eval.Add(ctb, temp, ctb)
			}
		}
		elp := time.Since(stt)
		fmt.Println("rotated input precomputation : ", i3, " ", elp)
		stt = time.Now()

		eval.RescaleTo(ctb, initScale, ctb)
		elp = time.Since(stt)
		fmt.Println("recaling : ", i3, " ", elp)
		stt = time.Now()

		inter := ctb.CopyNew()
		var sum *rlwe.Ciphertext
		var temp *rlwe.Ciphertext
		// summation for all input channels
		d, c := log2Int(ki), log2Int(ti)
		for x := 0; x < d; x++ {
			temp = inter.CopyNew()
			eval.Rotate(temp, pow2(x), temp)
			// RotatePrintCIFAR10Convolution(temp, pow2(x), temp, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
			eval.Add(inter, temp, inter)
		}
		for x := 0; x < d; x++ {
			temp = inter.CopyNew()
			eval.Rotate(temp, pow2(x)*ki*wi, temp)
			// RotatePrintCIFAR10Convolution(temp, pow2(x)*ki*wi, temp, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
			eval.Add(inter, temp, inter)
		}
		if c == -1 {
			sum = ctZero.CopyNew()
			for x := 0; x < ti; x++ {
				temp = inter.CopyNew()
				eval.Rotate(temp, ki*ki*hi*wi*x, temp)
				// RotatePrintCIFAR10Convolution(temp, ki*ki*hi*wi*x, temp, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
				eval.Add(sum, temp, sum)
			}
			inter = sum.CopyNew()
		} else {
			for x := 0; x < c; x++ {
				temp = inter.CopyNew()
				eval.Rotate(inter, pow2(x)*ki*ki*hi*wi, inter)
				// RotatePrintCIFAR10Convolution(inter, pow2(x)*ki*ki*hi*wi, inter, &eval, ki, hi, wi, ci, ti, pi, co, st, fh, fw, logn)
				eval.Add(inter, temp, inter)
			}
		}
		ctc := inter.CopyNew()
		elp = time.Since(stt)
		fmt.Println("sumslot : ", i3, " ", elp)
		stt = time.Now()

		for i4 := 0; (i4 < pi) && (i4+pi*i3 < co); i4++ {
			i := pi*i3 + i4
			temp, _ := eval.RotateNew(ctc, -(i/(ko*ko))*ko*ko*ho*wo+(n/pi)*(i%pi)-((i%(ko*ko))/ko)*ko*wo-(i%ko))
			eval.Mul(temp, selectOneVec[i], temp)
			eval.Add(ctd, temp, ctd)
		}
		elp = time.Since(stt)
		fmt.Println("result : ", i3, " ", elp)
	}
	elapse = time.Since(starttime)
	fmt.Println("total q : ", elapse)
	starttime = time.Now()

	eval.RescaleTo(ctd, initScale, ctd)
	result := ctd.CopyNew()
	for j := 0; j < int(log2IntPlusToll(float64(po))); j++ {
		temp, _ := eval.RotateNew(result, -pow2(j)*(n/po))
		eval.Add(result, temp, result)
	}
	elapse = time.Since(starttime)
	fmt.Println("multiplexing : ", elapse)

	cnnOut = TensorCipher{
		k_:      ko,
		h_:      ho,
		w_:      wo,
		c_:      co,
		t_:      to,
		p_:      po,
		logn_:   logn,
		cipher_: result,
	}
	return cnnOut
}
