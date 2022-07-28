
## 前言
第一個如果我們今天有很多很多的特徵轉換 要使用的時候，我們怎麼好好地運用這些特徵轉換？ 更重要的是，這麼多的特徵轉換，可能會有複雜度的問題，我們怎麼控制這些複雜度的問 題。那這樣的想法，刺激了一個很有名的模型，叫做 Support Vector Machine， 支撐向量機的發展。

## [1-1 Large-Margin Separating Hyperplane](https://www.youtube.com/watch?v=8hak0XngnV0&list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2kpk1U2&index=3)
### 哪一條線較好？為何最右邊？
透過測量誤差容忍度我們可以得到下面三種不同的線，就以 VC bound 來看，這三條線好像沒什麼差，其共通點都可以在訓練資料被完美的分開。如最右邊圖所示，如果我們的每一筆訓練資料跟線的距離，隔得越遠的話，那就表示我們的線，可以忍受越多的測量的誤差。因此我們希望訓練資料能夠離這些線越遠越好，這樣它就能夠容忍比較多的雜訊，同時能夠避免 overfitting 的情況發生。

![](https://i.imgur.com/X5j5btr.png)

> 定義一條線到底強不強壯就是，觀察離最接近的訓練資料到底有多遠。

從另一個角度來看就是觀察一條線能夠長到多胖，就是看從線往外面長出去，長到哪裡會碰到最接近的那些點。

![](https://i.imgur.com/BXnkTDH.png)

> 換句話說定義一條線到底強不強壯就是，就是比較胖的線。

因此我們想要找出來某一條線，這條線要滿足兩個特性。一個是跟 PLA 找出來的線一樣，必須將所有的訓練資料圈圈叉叉都好好的分開。接著在把它分開以後，在那麽多條把它分開的線裏面，我們需要選擇一條最胖的線(邊界最大)，就是最強壯最強固的線。至於該如何定義它到底有多胖呢？最簡單的方式就是把這條線與每一個點的距離都算一算，取最小的距離作為判斷依據。

![](https://i.imgur.com/ATtROj3.png)

- separating hyperplane
- find largest-margin

## 小試身手
### Question 1.
Consider two examples (v, +1) and (−v,−1) where v ∈ℝ2 (without padding the v_0 = 1v 
0=1). Which of the following hyperplane is the largest-margin separating one for the two examples? You are highly encouraged to visualize by considering, for instance, v =(3,2).

- A) x1​=0
- B) x2=0
- C) v1​x1​+v2​x2​=0
- D) v2​x1​+v1​x2​=0

> Ans: C

- v 是一個二維向量 $(v_1,v_2)$，資料點$(v_1,v_2)$ 是+1，點$(-v_1,-v_2)$ 是-1，找出可以分開這兩個資料點並且有最大margin的 Hyperplane
- 就是求 $v$ 和 $-v$ 的中垂線，即 $v_1x_1+v_2x_2=0$

## [1.2 Standard Large-Margin Problem](https://www.youtube.com/watch?v=lHo9GcIURRs&list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2kpk1U2&index=3)
### 如何計算 Margin？
接下來我們要探討 margin 的計算，該如何在超平面下計算距離。如下圖所示 x'' 與 x' 兩個都是平面上的點，所以它代表這樣是一串 平面上的向量。w 乘上平面上任何一個向量都等於 0，這代表 w 會垂直於這個平面。也就是說 w 是我們想象那個平面的法向量。既然如此，那我要求 $x$ 到 hyperplane 的距離，其實就是 $x$ 到 hyperplane 上任意點 $x'$ 的向量，投影到垂直平面的方向 $w$ 上即可。

![](https://i.imgur.com/nncpbKh.png)

- 若有任意兩個點 $x',x''$ 都在 hyperplane $w^Tx+b=0$ 中
- 我們知道 $w^Tx'=-b$ 且 $w^Tx''=-b$
- 所以 $w^T(x''-x')=0$
    - 也就是說 vector on hyperplane 跟 $w$ 的內積會是 0
    - 所以 **$w$ 向量垂直於這個平面，也就是 hyperplane 的法向量**
- 既然如此，那我要求 $x$ 到 hyperplane 的距離，其實就是 $x$ 到 hyperplane 上任意點 $x'$ 的向量，投影到 $w$ 上即可。
- $distance(x,b,w)=|\dfrac{w^T}{\|w\|}(x-x')|=\dfrac{1}{\|w\|}|w^Tx+b|$
    - 只要回憶一下向量投影長公式 ($v_1$ 投影到向量 $v_2$) $\dfrac{v_1\cdot v_2}{\|v_2\|}=\dfrac{v_1^Tv_2}{\|v_2\|}=\dfrac{v_2^Tv_1}{\|v_2\|}$ 並利用 $w^Tx'=-b$ 即可得證。
- 注意要除以這個方向上面的 w 長度，也就是說投影過去，要看看那個方向上單位的長度是怎麼樣。

### 僅計算可分離超平面的距離
目前已經學會給定一個平面或線可以計算點到這條線或平面的距離。但是我們今天考慮的不是任何一條線，而是可以把圈圈跟叉叉完美的分開的線，叫做分隔線 Separating Hyperplane。也就是就是我們算出來的分數，跟我們想要的圈圈叉叉要是同號的(兩個相乘極大於0)。這個式子即可以取代原有的 $|w^Tx+b|$。我們現在已經知道怎麼算了，算出分數來乘上相對應的 y 然後除以這個 w 的長度。因為在我們的條件裡我們只考慮能夠把所有的圈圈叉叉都分對的這些線。

![](https://i.imgur.com/PGubQbU.png)

### 更進一步簡化式子
想一下若最小一個剛好等於 1，那 margin 就可以直接寫下最小的等於1乘上1除以這個 w 長度，所以我的 margin，就可以用一個很簡單的式子寫出來了，1除上 w 長度。

![](https://i.imgur.com/rxjVqQ3.png)

- 既然對 $w,b$ 同時做縮放，不會影響 hyperplane，那麼乾脆讓最靠近 hyperplane 的點 $n$ 的 $y_n(w^Tx_n+b)=1$
- 這樣子我們的 margin 就可以簡化成 $\frac 1 {\|w\|}$ 
- 不過這樣就多一個條件，就是 $\min_\limits{n=1,...,N}y_n(w^Tx_n+b)=1$
    - 有了這個條件，之前的條件 $y_n(w^Tx_n+b)>0,\forall (x_n,y_n)\in\mathcal D$ 一定會成立，所以不用再特別寫了。
- optimization problem 現在變成：
    - 目標 $\max_\limits{b,w} \frac 1{\|w\|}$
    - 條件 $\min_\limits{n=1,...,N}y_n(w^Tx_n+b)=1$

所以我們式子現在變成下圖所示，我們要想辦法把這個東西做得越大越好。這個東西是我們現在的 margin 我們加上的條件是裡面最小的 y 乘上分數要剛好等於 1。

![](https://i.imgur.com/9RbVtwQ.png)

![](https://i.imgur.com/vp8nrst.png)

## [1.3 Support Vector Machine](https://www.youtube.com/watch?v=FAm70y081o4&list=PLXVfgk9fNX2IQOYPmqjqWsNUFl2kpk1U2&index=4)

## 重點整理
- 我們要找出的一條線需要與每個點都越遠越好，使得可以容忍較大的測量誤差。
- 定義一條線到底有多強壯就是，一條線的邊界有多胖，或者觀察離直線最近的點距離是多少？
