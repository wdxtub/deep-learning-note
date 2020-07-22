// https://leetcode-cn.com/problems/3sum/

func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	var result [][]int
	for i := 0; i < len(nums) -2; i++ {
		s := nums[i]
		if s > 0 {
			break
		}
		p1, p2 := i+1, len(nums) -1
		for p1 < p2 {
			sum := s + nums[p1] + nums[p2]
			if sum < 0 {
				p1 += 1
			} else if sum > 0 {
				p2 -= 1
			} else {
				result = append(result, []int{s, nums[p1], nums[p2]})
				for p1 < p2 && nums[p1] == nums[p1+1] {
					p1 += 1
				}
				for p1 < p2 && nums[p2] == nums[p2-1] {
					p2 -= 1
				}
				p1 += 1
				p2 -= 1
			}
		}
		for i < len(nums)-3 && nums[i] == nums[i+1] {
			i += 1
		}
	}
	return result
}