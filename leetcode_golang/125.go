// https://leetcode-cn.com/problems/valid-palindrome/

func isPalindrome(s string) bool {
	p1 := 0
	p2 := len(s) -1
    pivot := len(s)-1

	for p1 < p2{
		for !validChar(s[p1]) && p1 < pivot {
			p1++
		}
		for !validChar(s[p2]) && p2 > 0{
			p2--
		}
        if p1 >= p2 {
            break
        }
		if !(s[p1] == s[p2] || 
        (s[p1] == s[p2] + 32 && s[p2] >= 'A') || 
        (s[p1] + 32 == s[p2] && s[p1] >= 'A') ){
			return false
		}
		p1++
		p2--
	}
	return true
}

func validChar(char byte) bool {
	if char >= '0' && char <= '9' {
		return true
	}
	if char >= 'A' && char <= 'Z' {
		return true
	}
	if char >= 'a' && char <= 'z' {
		return true
	}
	return false
} 