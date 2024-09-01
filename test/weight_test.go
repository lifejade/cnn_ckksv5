package test

import (
	"fmt"
	"testing"
)

func Test_filter_weight(t *testing.T) {

	fh, fw, co := 3, 3, 32
	hi, wi, ci, ki, ti, pi := 16, 16, 32, 2, 8, 4
	q, n := (co+pi-1)/pi, 1<<15

	data := make([]float64, n)
	for i, _ := range data {
		data[i] = float64(i)
	}
	_ = filter_weight_my

	my := filter_weight_realmy(fh, fw, hi, wi, ci, co, ki, ti, pi, q, n, data)
	pro := filter_weight_pro(fh, fw, hi, wi, ci, co, ki, ti, pi, q, n, data)
	for i1, v1 := range my {
		for i2, v2 := range v1 {
			for i3, v3 := range v2 {
				for i4, v4 := range v3 {
					if pro[i1][i2][i3][i4] != v4 {
						t.Errorf("%d %d %d %d\n %f %f \n", i1, i2, i3, i4, pro[i1][i2][i3][i4], v4)
						return
					}
				}
			}
		}
	}
}

func filter_weight_realmy(fh, fw, hi, wi, ci, co, ki, ti, pi, q, n int, data []float64) [][][][]float64 {
	get_weight := func(h, w, i, o int) float64 {
		return data[fh*fw*ci*o+fh*fw*i+fw*h+w]
	}

	//multiplex parallel packing fillter
	weight := make([][][][]float64, fh)
	for i1 := 0; i1 < fh; i1++ {
		weight[i1] = make([][][]float64, fw)
		for i2 := 0; i2 < fw; i2++ {
			weight[i1][i2] = make([][]float64, q)
			for i3 := 0; i3 < q; i3++ {
				weight[i1][i2][i3] = make([]float64, n)
				for i := 0; i < n; i++ {
					temp := i
					i6 := temp % (ki * wi)
					temp /= (ki * wi)
					i5 := temp % (ki * hi)
					temp /= (ki * hi)
					i7 := temp
					var pow2fit_ti int = n / (ki * ki * wi * hi * pi)
					if (i7%pow2fit_ti) >= ti || ki*ki*(i7%pow2fit_ti)+ki*(i5%ki)+(i6%ki) >= ci || (i7/pow2fit_ti)+pi*i3 >= co ||
						(i5/ki)-(fh-1)/2+i1 < 0 || (i5/ki)-(fh-1)/2+i1 > hi-1 || (i6/ki)-(fw-1)/2+i2 < 0 || (i6/ki)-(fw-1)/2+i2 > wi-1 {
						weight[i1][i2][i3][i] = 0
					} else {
						weight[i1][i2][i3][i] = get_weight(i1, i2, ki*ki*(i7%pow2fit_ti)+ki*(i5%ki)+(i6%ki), pi*i3+(i7/pow2fit_ti))
					}
				}
			}
		}
	}
	return weight
}

func filter_weight_pro(fh, fw, hi, wi, ci, co, ki, ti, pi, q, n int, data []float64) [][][][]float64 {
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
	return compactWeightVec
}

func filter_weight_my(fh, fw, hi, wi, ci, co, ki, ti, pi, q, n int, data []float64) [][][][]float64 {
	get_weight := func(h, w, i, o int) float64 {
		return data[fh*fw*ci*o+fh*fw*i+fw*h+w]
	}

	//multiplex parallel packing fillter
	weight := make([][][][]float64, fh)
	for i1 := 0; i1 < fh; i1++ {
		weight[i1] = make([][][]float64, fw)
		for i2 := 0; i2 < fw; i2++ {
			weight[i1][i2] = make([][]float64, q)
			for i3 := 0; i3 < q; i3++ {
				weight[i1][i2][i3] = make([]float64, n)
				for i := 0; i < n; i++ {
					if i == 3105 {
						fmt.Printf("")
					}
					size := n / pi
					i5 := (i % size) % (ki * ki * hi * wi) / (ki * wi)
					i6 := (i % size) % (ki * wi)
					i7 := (i % size) / (ki * ki * wi * hi)
					if ki*ki*(i7)+ki*(i5%ki)+(i6%ki) >= ci || (i7/ti)+pi*i3 >= co ||
						(i5/ki)-(fh-1)/2+i1 < 0 || (i5/ki)-(fh-1)/2+i1 > hi-1 || (i6/ki)-(fw-1)/2+i2 < 0 || (i6/ki)-(fw-1)/2+i2 > wi-1 {
						weight[i1][i2][i3][i] = 0
					} else {
						weight[i1][i2][i3][i] = get_weight(i1, i2, ki*ki*(i7)+ki*(i5%ki)+(i6%ki), pi*i3+(i/size))
					}
				}
			}
		}
	}
	return weight
}
