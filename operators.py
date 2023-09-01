import numpy as np
#np.warnings.filterwarnings('ignore')


def lag(series, periods=1):
    '''
    將序列值落後一期。
    '''
    series = np.roll(series, shift=periods)
    # np.roll會把最後的值塞到最前面，因此要把前面值變成nan
    #[: periods] 從第0到第periods個索引
    series[: periods]= np.nan
    return series

# 此operator 是處理連續增長或衰退的情況
def lag_series_for_change_in_a_row(series,periods,type = 'p'):
    if periods > 1:
        # 如果periods 給2，表示連續2季成長
        # 後面range(1,2) 就只會跑一圈 i=1
        periods = periods + 1
        # 如果是連續成長的狀態
        if type == 'p':
            # 初始化 result numpy series 全部都是True 長度跟 series一樣
            #
            result = np.full_like(series, True, dtype=bool)
            
            for i in range(1,periods):
                #每次迴圈都會把原始的series往後移動一格觀察
                temp_series = np.roll(series, shift= 1)
                temp_series[: i] = np.nan
                
                # 紀錄比較後的結果，只要迴圈裡某一次不符合條件就會給False
                result = (series >= temp_series ) & (result) 

                # 將series迭代繼續跟temp_series比較
                series = temp_series
        # 如果是連續衰退狀態   
    else:
        print("只用lag函數就可以了")
    
    return result

def lag_series_more_careful(series,periods):
    
    # 初始化 result numpy series 全部都是True 長度跟 series一樣
    result = np.full_like(series, False, dtype=bool)
    
    #迴圈跑整個series一一檢查, 從欲觀察的 index 開始檢查
    # periods = 2 就是從第2個index去看前面兩個有沒有連續成長
    for i in range(0 + periods, len(series)):
        # 先取出欲觀察的series 長度, periods =2 會拿到長度是3的series
        temp_series = series[:i+1]
        
        # flag 是要去看有沒有真的滿足數據要求的成長數量
        # 例如 20 23 23 ,在23往回看的時候，第一次(23,23)沒有成長所以flag不增加
        # 但是第二次 (23,20)有比較大所以flag+1
        flag = 0
        # ex: periods =2 一開始從倒數第一個index 去檢查到第0個, len(temp_series) =3
        # 迴圈從欲觀察的index 跑到整個series的頭 (倒過來檢查)
        # range(1,3) 會跑兩圈
        
        for j in range(1 , len(temp_series)):
            # 如果滿足條件了
            if flag == periods:
                result[i] = True
                break
            # 如果倒數第j個 index 值 < 倒數第j-1個 index值:
            # 直接不符合條件 break
            elif (temp_series[-j] < temp_series[-j-1]) or (np.isnan(temp_series[-j] - temp_series[-j-1]).any()):
                # result[i] = False
                break
            # 如果倒數第j個 index 值 > 倒數第j-1個 index值:
            elif temp_series[-j] > temp_series[-j-1]:
                flag = flag + 1
            # 如果數值兩個值比較是一樣維持不變，則flag不變繼續迴圈(往前看)
            elif temp_series[-j] == temp_series[-j-1]:
                continue
            # 如果要比較的數值有 Nan 則直接是 False, break迴圈
            # elif np.isnan(temp_series[-j] - temp_series[-j-1]).any():
            #     break
            

            
    return result
    
    

def rolling_window(series, windows):
    '''
    模仿pandas.rolling用法，但底層是由numpy形成。
    (運作邏輯可參考以下網址)
    '''
    # 參考https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
    shape = series.shape[:-1] + (series.shape[-1] - windows + 1, windows)
    strides = series.strides + (series.strides[-1],)
    
    # 在上述方法下，將只會算出移動後的情況，例如：series的len為5，移動窗格為3
    # series為[1, 2, 3, 4, 5]，經由上面做法會產出
    # [
    #  [1, 2, 3],
    #  [2, 3, 4],
    #  [3, 4, 5]
    # ]
    # 但實際序列有5個值因此以下在前面設定nan
    # [
    #  [nan, nan, nan],
    #  [nan, nan, nan]
    # ]
    # 以下依據上述邏輯，產生兩個array後再疊一起
    # nan的shape size為(windows-1, windows)
    # 其中x為windows-1，因為第windows個時便可以開始運算，所以只有windows-1
    # Y由於給定窗格為大小為windiws，因此為windiws
    na_array = np.empty(shape=(windows-1, windows))
    na_array[:] = np.nan
    
    rolled_array = np.lib.stride_tricks.as_strided(
        series, shape=shape, strides=strides
    )
    
    result = np.vstack((na_array, rolled_array))
    
    return result