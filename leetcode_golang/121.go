// https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/

func maxProfit(prices []int) int {
	sofarMin := -1
	profit := 0
	for i := 0 ;i < len(prices); i++ {
		if sofarMin == -1 {
			sofarMin = prices[i]
		} else {
			if prices[i] - sofarMin > profit {
				profit = prices[i] - sofarMin
			}
			if sofarMin > prices[i] {
				sofarMin = prices[i]
			}
		}
	}
	return profit
}	