package comp

import (
	"fmt"
	"math"
)

type Tree struct {
	depth int
	tree  []int
	m     int
	l     int
	b     int
}

func NewTree() *Tree {

	tree := &Tree{
		depth: 0,
		m:     0,
		l:     0,
		b:     0,
	}

	(*tree).tree = make([]int, 2)
	(*tree).tree[0] = -1
	(*tree).tree[1] = 0

	return tree
}
func (tree *Tree) Copy(cpTree Tree) {

	// cpTree : unchanged
	tree.depth = cpTree.depth
	tree.m = cpTree.m
	tree.l = cpTree.l
	tree.b = cpTree.b
	tree.tree = make([]int, len(cpTree.tree))
	copy(tree.tree, cpTree.tree)

}
func (tree *Tree) Clear() {

	tree.depth = 0
	tree.tree = make([]int, 2)
	tree.tree[0] = -1
	tree.tree[1] = 0
}
func (tree *Tree) Print() {

	fmt.Println("depth of tree: ", tree.depth)
	for i := 0; i <= tree.depth; i++ {
		for j := Pow2(i); j < Pow2(i+1); j++ {
			fmt.Print(tree.tree[j], " ")
		}
		fmt.Println()
	}

	fmt.Println("m: ", tree.m)
	fmt.Println("l: ", tree.l)

	nonscalar := tree.m - 1 + Pow2(tree.l-1) - 1
	for i := 0; i < Pow2(tree.depth+1); i++ {
		if tree.tree[i] > 0 {
			nonscalar++
		}
	}
	fmt.Println("nonscalar: ", nonscalar)
	fmt.Println()

}

func (tree *Tree) Merge(a, b *Tree, g int) {

	tree.Clear()
	if a.depth > b.depth {
		tree.depth = a.depth + 1
	} else {
		tree.depth = b.depth + 1
	}

	tree.tree = make([]int, Pow2(tree.depth+1))
	for i := 0; i < Pow2(tree.depth+1); i++ {
		tree.tree[i] = -1
	}
	tree.tree[1] = g

	for i := 1; i <= Pow2(a.depth+1)-1; i++ {

		temp := Pow2(int(math.Log2(float64(i))))
		tree.tree[i+temp] = a.tree[i]
	}
	for i := 1; i <= Pow2(b.depth+1)-1; i++ {

		temp := Pow2(int(math.Log2(float64(i))))
		tree.tree[i+2*temp] = b.tree[i]
	}
}
func OddBaby(n int) (tree *Tree) {

	d := int(math.Ceil(math.Log2(float64(n))))
	totalMin, minm, minl := 10000, 0, 0
	var totalMinTree *Tree

	for l := 1; Pow2(l)-1 <= n; l++ {
		for m := 1; Pow2(m-1) < n; m++ {

			// initialization
			f := make([][]int, n+1)
			for i := 0; i < n+1; i++ {
				f[i] = make([]int, d+1)
			}
			G := make([][]*Tree, n+1)
			for i := 0; i < n+1; i++ {
				G[i] = make([]*Tree, d+1)
				for j := 0; j < d+1; j++ {
					G[i][j] = NewTree()
				}
			}
			f[1][1] = 0
			for i := 3; i <= n; i += 2 {
				f[i][1] = 10000
			}

			// recursion
			for j := 2; j <= d; j++ {
				for i := 1; i <= n; i += 2 {
					if i <= Pow2(l)-1 && i <= Pow2(j-1) {
						f[i][j] = 0
					} else {
						min := 10000
						minTree := NewTree()
						for k := 1; k <= m-1 && Pow2(k) < i && k < j; k++ {
							g := Pow2(k)
							if f[i-g][j-1]+f[g-1][j]+1 < min {
								min = f[i-g][j-1] + f[g-1][j] + 1
								minTree.Merge(G[g-1][j], G[i-g][j-1], g)
							}
						}
						f[i][j] = min
						G[i][j] = minTree

					}

				}
			}

			if f[n][d]+Pow2(l-1)+m-2 < totalMin {
				totalMin = f[n][d] + Pow2(l-1) + m - 2
				totalMinTree = G[n][d]
				minm = m
				minl = l
			}

		}
	}

	// fmt.Println("deg ", n, ": ", totalMin)
	// fmt.Println("m ", minm, ", l: ", minl)
	tree = totalMinTree
	tree.m = minm
	tree.l = minl

	return tree
}
