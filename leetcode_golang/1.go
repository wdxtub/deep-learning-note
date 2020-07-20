// https://leetcode-cn.com/problems/two-sum/

func twoSum(nums []int, target int) []int {
	l := len(nums)
	result := []int{}
	finish := false
	for i := 0; i < l; i++ {
		for j := i+1; j < l; j++ {
			if nums[i] + nums[j] == target {
				result = append(result, i)
				result = append(result, j)
				finish = true
				break
			}
		}
		if finish {
			break
		}
	}
	return result
}