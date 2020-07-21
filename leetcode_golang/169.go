// https://leetcode-cn.com/problems/majority-element/

func majorityElement(nums []int) int {
	length := len(nums)
	if length == 1 {
		return nums[0]
	}

	num_dict := map[int]int{}
    common := 0
	for i := 0 ; i < length; i++ {
		item := nums[i]
		if _, ok := num_dict[item]; ok {
			num_dict[item] += 1
			if num_dict[item] > length/2 {
				common = item
                break
			} 
		} else {
			num_dict[item] = 1
		}
	}
    return common
}