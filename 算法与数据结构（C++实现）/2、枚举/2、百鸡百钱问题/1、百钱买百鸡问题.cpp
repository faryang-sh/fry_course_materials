
/*

百钱买百鸡
公鸡一只五块钱，母鸡一只三块钱，小鸡一块钱三只，
现在要用一百块钱买一百只鸡，每种鸡最少一只，问公鸡、母鸡、小鸡各多少只？


枚举变量：公鸡、母鸡、小鸡
枚举范围：公鸡、母鸡、小鸡都是1-100次，总计算次数100*100*100
枚举判断条件：
钱数=100：5公鸡+3母鸡+1/3小鸡 = 100
总鸡数=100：公鸡+母鸡+小鸡 = 100
小鸡%3==0

*/

#include <stdio.h>
int main()
{
    for (int i = 1; i <= 100; i++)
        for (int j = 1; j <= 100; j++)
            for (int k = 1; k <= 100; k++)
            {
                if (5 * i + 3 * j + k / 3 == 100 && k % 3 == 0 && i + j + k == 100)
                {
                    printf("公鸡 %2d 只，母鸡 %2d 只，小鸡 %2d 只\n", i, j, k);
                }
            }
    return 0;
}
