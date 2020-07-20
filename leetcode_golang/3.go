// https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

func lengthOfLongestSubstring(s string) int {
	max := 0
	start := -1
	order := map[byte]int{}
	if len(s) == 1 {
		return 1
	}
	for i := 0; i < len(s); i++ {
		if _, ok := order[s[i]]; ok { // exists
			if start < order[s[i]] {
				start = order[s[i]]
			}
			order[s[i]] = i
			if i - start > max {
				max = i - start
			}
		} else { // noexists
			order[s[i]] = i
			if i - start > max {
				max = i - start
			}
		}
	}
	return max
}