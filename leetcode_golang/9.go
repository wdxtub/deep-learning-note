// https://leetcode-cn.com/problems/palindrome-number/

func isPalindrome(x int) bool {
	if x < 0 || (x % 10 == 0 && x != 0){
		return false
	}
	reverse := 0
	remain := 0
	for {
		remain = x % 10
		x /= 10
		reverse = reverse * 10 + remain
		if reverse >= x {
			break
		}
	}
	return reverse == x || reverse/10 == x
}