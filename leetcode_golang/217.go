// https://leetcode-cn.com/problems/contains-duplicate/

func containsDuplicate(nums []int) bool {
	freq := map[int]int{}
	for i := 0; i < len(nums); i++ {
		if _, ok := freq[nums[i]]; ok {
			return true
		} 
		freq[nums[i]] = 1
	}
	return false
}