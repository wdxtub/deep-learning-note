// https://leetcode-cn.com/problems/climbing-stairs/

func climbStairs(n int) int {
	a := 0
	b := 0
	c := 1
	for i := 1; i <= n; i++ {
		a = b
		b = c
		c = a + b
	}
	return c
}
