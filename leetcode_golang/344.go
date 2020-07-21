// https://leetcode-cn.com/problems/reverse-string/

func reverseString(s []byte)  {
	end := len(s) - 1
	start := 0
	for {
		if start >= end {
			break
		}
		tmp := s[start]
		s[start] = s[end]
		s[end] = tmp
		start += 1
		end -= 1
	}
}