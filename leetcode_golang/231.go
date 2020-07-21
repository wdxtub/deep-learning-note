// https://leetcode-cn.com/problems/power-of-two/

func isPowerOfTwo(n int) bool {
	if n <= 0 {
        return false
    }
    if n < 3 {
		return true
	}

	for {
		remain := n >> 1
		if remain * 2 != n {
			return false
		}
		n = remain
		if n < 3 {
			break
		}
	}
	return true
}