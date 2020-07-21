// https://leetcode-cn.com/problems/nim-game/

func canWinNim(n int) bool {
    if n % 4 == 0 {
        return false
    }
    return true
}