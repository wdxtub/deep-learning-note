func threeSumClosest(nums []int, target int) int {
	sort.Ints(nums)
	n := len(nums)
	best := math.MaxInt32

	for i := 0; i < n; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}

		p1, p2 := i+1, n-1
		for p1 < p2 {
			sum := nums[i] + nums[p1] + nums[p2]
			if sum == target {
				return target
			}
			if abs(sum - target) < abs(best - target) {
				best = sum
			}
			if sum > target {
				for p1 < p2 && nums[p2] == nums[p2-1] {
					p2 -= 1
				}
				p2 -= 1
			}
			if sum < target {
				for p1 < p2 && nums[p1] == nums[p1+1] {
					p1 += 1
				}
				p1 += 1
			}
		}
	}
	return best
}

func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}