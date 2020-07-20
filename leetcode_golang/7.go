// https://leetcode-cn.com/problems/reverse-integer/

func reverse(x int) int {
	MAX := 1 << 31 - 1
	MIN := -1* MAX -1
	result := 0
    flag := 1
    if x < 0 {
        x *= -1
        flag = -1
    }
	for {
		if x > 0 {
			t := x % 10
			x /= 10
			if result > MAX/10 || (result == MAX/10 && t > 7) {
				return 0
			}
			if result < MIN/10 || (result == MIN/10 && t < -8) {
				return 0
			}
			result = result * 10 + t
		} else {
			break
		}
	}
	return result*flag
}