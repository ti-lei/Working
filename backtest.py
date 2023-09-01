from itertools import compress
import numpy as np
import pandas as pd
from preprocess import Data
from datetime import datetime
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA

class Backtest:
    def __init__(
        self,
        signal_list,
        report_mode=False,
        long_or_short='long',
        start_date='2005/1/1',
        end_date='2019/7/5'
    ):
        '''
        在Data帶有訊號後，放入signal_list中，開始執行回測。

        Parameters
        ----------
        signal_list:
            輸入訊號源
        report_mode：
            選擇欲使用之價格資料同時決定進出場時點。
            - False：使用開盤價，並以訊號產生後下一個開盤價進出場
            - True：使用收盤價，並以訊號產生當個收盤價進出場
            (使用True時會跳出警告，因為此價格基本上不可能成交)
        long_or_short：
            針對產出之訊號決定做多或是放空。
            - long：做多
            - short：放空
        start_date:
            回測起始日
        end_date:
            回測結束日
            (可能會出現最後一期持有天數很短的現象，因為在正常的出場日期到來前，
            就因設定日期而提前出場)
        
        Notes：
        ------
        1. 不論是修改策略或是更改回測日期，皆必須先執行重新加入filter的程式，
           因為執行.run()中處理signal的部分會把原先signal帶有的dict格式改掉，
           若沒有重新加入filter產生原先signal的dict格式，將會出現error
        2. 當初將變數命名為report_mode的目的在於要寫報告時需要較及時的結果，
           例如：若要在周五下午依據財務資訊產出報告，但使用次一個開盤價進出場會需要
           等到下周一開盤才能知道績效結果，因此以產出報告的角度太慢，所以新增此report_mode
           提供選擇，但需留意report_mode=True的模式下，以實際交易而言是較不真實的。
        '''
        # 將signal list依據頻率由低至高排序，後面帶入for loop需要低頻至高頻
        # x._Data__data_frequency_num為讀取Data class內部屬性frequency_num之方法

        self.signal_list = sorted(signal_list, key=lambda x: x._Data__data_frequency_num)

        # 由於signal_list已經由低頻排至高頻，且最後訊號將會依據最高頻進行篩選
        # 在Analysis模組中會使用到qualified_market_size_signal
        # 但signal_list中每個訊號都帶有自己的qualified_market_size_signal
        # 在上述架構下，由於最終會依據最高頻者進行考量，因此使用排序後signal_list的最後一者
        self.qualified_market_size_signal = self.signal_list[-1].qualified_market_size_signal
        self.start_date = datetime.strptime(start_date, '%Y/%m/%d')
        self.end_date = datetime.strptime(end_date, '%Y/%m/%d')

        self.report_mode = report_mode
        # 確保long_or_short參數正確
        if long_or_short not in ['long', 'short']:
            raise Exception('未知的long_or_short，請選擇long或short')
        if long_or_short == 'short':
            print('Warning：放空模式已開啟，訊號觸發後將執行放空！')
        self.long_or_short = long_or_short
        # 依據給定report_mode讀入欲使用資料價格並命名為trading_price
        # 用以判斷實際交易時點與進出場使用之價格
        if self.report_mode == False:
            self.trading_price = Data('open_price', data_frequency='day')
        elif self.report_mode == True:
            print('Warning：report_mode已開啟，訊號觸發後將使用當個收盤價進出場！')
            print('')
            self.trading_price =Data('close_price', data_frequency='day')
        else:
            raise Exception('未知的report_mode，請選擇True或False')
        print('stop')
    def run(self):
        '''
        開始進行回測
        '''
        # 建立選擇指定時間區間的functuion
        def select_interval(strategy):
            signal = {}
            for key in strategy.signal.keys():
                if (key >= self.start_date) & (key <= self.end_date):
                    signal[key] = strategy.signal[key]
            return signal

        # 依據給定回測期間限縮欲使用價格之時間區間
        # 於源頭處限縮區間，意謂最後一筆期交易時間可能會出現持有天數很短的現象
        # 因為在真正出場日期到來前便因設定時間區間到期而出場
        self.trading_price.data = self.trading_price.data.iloc[
            :, (
                (self.trading_price.data.columns >= self.start_date) &
                (self.trading_price.data.columns <= self.end_date)
            )
        ]

        # 在jupyter notebook跑時，若想要在不restart kernal的情況下重新定義策略
        # 在run執行前就必須先確保stock_screened_by_strategy不存在
        # 因為若有執行過，stock_screened_by_strategy便會存在
        # 因此在執行run前先檢查stock_screened_by_strategy是否存在
        # 若存在則將其刪除，便可以不用每次要重新跑策略還要restart kernal
        if 'stock_screened_by_strategy' in locals():
            del stock_screened_by_strategy

        # 由低頻處理至高頻，因此先處理非日頻率之資料
        for strategy in self.signal_list:
            strategy.signal = select_interval(strategy)
            # 原先startegy中的signal格式為{'Timestamp': {2330: '財務數值'}}
            # 由於被納入訊號中已代表符合條件可被交易，因此財務數值已不必再被使用
            # 因此以下只萃取各個Timestamp中的股票代碼(代碼為keys)
            # {'Timestamp': {2330: '財務數值'}} -> {'Timestamp': {'2330'}}
            tt = {}
            for key in strategy.signal.keys():
                strategy.signal[key] = set(strategy.signal[key].keys())
                tt[key] = sorted(strategy.signal[key])
            # tt 是為了觀察各策略的進出場所放上去的
            # 這時strategy.signal 已經整理成每個時間點 對應到的策略
            pd.DataFrame.from_dict(tt,orient='index'
            ).to_csv(strategy.data_name+'_strategy.csv')
            # 首先判斷'stock_screened_by_strategy'是否已建立
            # 若否則將最一開始接受到的signal(signal_list頻率最低者)指定為訊號(初始訊號)
            # 最低頻的 signal 會進到這個if裡面，之後相對高頻的訊號近來就都是走else
            # 'stock_screened_by_strategy'將會在後面被一直迭代
            
            if 'stock_screened_by_strategy' not in locals():
                # stock_screened_by_strategy: {'Timestamp': {2330, 1101...}}
                stock_screened_by_strategy = strategy.signal
            else:
                # 建立空的'temp_stock_screened_by_strategy'用來暫時裝第二次迴圈產生的內容
                temp_stock_screened_by_strategy = {}
                
                # for 每個 timestamp,走這條路的時候 stock_screened_by_strategy是上一個signal的東西
                for i in range(len(stock_screened_by_strategy)):
                    date_entry = list(stock_screened_by_strategy.keys())[i]

                    # 此處用try的目的在於當i為最後一個值時若出現i+1則會有
                    # 'list index out of range'的error，此時則會進到except
                    # 便直接指定date_exit為既有交易資料日期中的最後一筆
                    try:
                        date_exit = list(stock_screened_by_strategy.keys())[i+1]
                    except:
                        date_exit = self.trading_price.data.columns[-1]
                    # 處理低頻率訊號
                    # main_tradable_set 第一次跑就是最低頻的訊號
                    
                    main_tradable_set = stock_screened_by_strategy[date_entry]

                    # 處理(較)高頻率訊號
                    # 由於訊號在signal_list中會依據低頻至高頻排序，因此也有可能為同頻率的狀況
                    # 由於可交易頻率為月以上(if)，所以日週期的策略另外處理(else)
                    if True:  # strategy.frequency != 'day'
                        # 季頻率資訊可能搭配月頻率資訊，因此兩季之間可能含有月資料
                        # 所以要找出落於date_entry, date_exit之間的可交易日與符合篩選條件之個股
                        # 例如同時使用季與月資料時，如5/15~8/14之間含有6/10, 7/10, 8/10之月資訊公告
                        # 此時main_tradable_set則為5/15當時，符合季篩選條件之個股清單
                        # 而sub_tradable_set則為各月營收公告時，符合月篩選條件之個股清單(以上述為例則有3份清單)
                        # 接者將main_tradable_set與sub_tradable_set取交集(結果會有三分個股清單)
                        # 但也可能是處理同頻率之資料，情況則沒有上述那麼複雜，純粹取同頻率之交集

                        # 找尋落於date_entry, date_exit之交易日
                        # 要把dict的keys轉成np.array後面才方便處理
                        strategy_signal_keys = np.array(list(strategy.signal.keys()))
                        index = (strategy_signal_keys >= date_entry) & (strategy_signal_keys <= date_exit)
                        # index中帶有True, False，compress會找出strategy.signal.keys()中對應到True者
                        trade_date = list(compress(strategy.signal.keys(), index))

                        for date in trade_date:
                            #sub_tradable_set:是股號
                            sub_tradable_set = strategy.signal[date]
                            # 取交集，找尋同時符合低頻率與高頻率者
                            temp_stock_screened_by_strategy[date] = main_tradable_set.intersection(
                                sub_tradable_set
                            )
                    else:
                        # 處理日資料
                        # 若日資料採行上述相同作法則會產生每日可交易股票清單(不符合我們需求)
                        # 但可交易頻率為月以上，因此需要另外處理日資料
                        # 直接找出高於日資料頻率的時間起始點
                        # 例如月資料公布日為6/10
                        # 則直接找出符合6/10之月篩選條件個股清單，且同時符合6/10當日篩選資料個股清單之交集
                        sub_tradable_set = strategy.signal[date_entry]
                        # 取交集，找尋同時符合低頻率與高頻率者
                        temp_stock_screened_by_strategy[date_entry] = main_tradable_set.intersection(
                            sub_tradable_set
                        )
                # 迭代stock_screened_by_strategy
                # 將temp_stock_screened_by_strategy指定回stock_screened_by_strategy
                # 並將temp_stock_screened_by_strategy清空    
                stock_screened_by_strategy = temp_stock_screened_by_strategy
                temp_stock_screened_by_strategy = {}
        
        # 每個時間點有訊號的進場股票
        self.entry_point = stock_screened_by_strategy

        # 產生實際交易表
        # 首先定義內部使用之function找尋距離公布日最近的當個或次個交易日
        def find_actual_entry_exit_date(date):
            # 透過report_mode判斷欲使用當個或次個價格進出場
            if self.report_mode == False:
                temp_index = self.trading_price.data.columns > date
            elif self.report_mode == True:
                temp_index = self.trading_price.data.columns >= date

            try:
                temp_target = self.trading_price.data.columns[temp_index]
                target = min(temp_target)
            # 若剛好為最後一期則temp_target為空值，取min則會出現error因此進到except
            # 此時直接指定target為現有的最後一筆交易日
            except:
                target = self.trading_price.data.columns[-1]
            return target

        # 使用progressbar顯示進度，以下設定相關outlay
        widgets = [
            'Progress: ', Percentage(), ' ',
            Bar('#'),' ', Timer(),
            ' , ', ETA(), ' '
        ]
        bar = ProgressBar(widgets=widgets, maxval=len(self.entry_point.keys())).start()
        trade_table = {}
        for i in range(len(self.entry_point.keys())):
            temp_trade_table = {}
            measure_date = list(self.entry_point.keys())[i]
            try:
                date_exit = list(self.entry_point.keys())[i+1]
            except:
                date_exit = self.trading_price.data.columns[-1]
            # 找尋實際交易日
            actual_date_entry = find_actual_entry_exit_date(measure_date)
            actual_date_exit = find_actual_entry_exit_date(date_exit)
            
            # 現在這個時間點裡滿足訊號的標的股票
            current_stock_entry_pool = self.entry_point[measure_date]
            
            # 找出進出場日之間的實際價格序列，並將日期與價格存起來
            # holding_period_price_interval 把所有股票的區間價格都記錄起來
            # 因為需要考量每日報酬狀況，所以要追蹤進出場日之間之損益
            holding_period_price_interval = self.trading_price.data.iloc[:, 
                (self.trading_price.data.columns >= actual_date_entry) &
                (self.trading_price.data.columns <= actual_date_exit) 
            ]
            # temp_trade_table: {'2330': {'2019/1/1': 200, '2019/1/2': 201...}}
            # 下面這裡的 zip 會把 key:時間點 對應的 value:價格 一個一個包成dicitionary
            # assign 給 temp_trade_table 時再包一層股號
            for code in current_stock_entry_pool:
                temp_trade_table[code] = dict(zip(
                    pd.to_datetime(
                        holding_period_price_interval[
                            holding_period_price_interval.index == code
                        ].columns
                    ),
                    holding_period_price_interval[
                        holding_period_price_interval.index == code
                    ].values.ravel()
                ))
            # 多儲存measure_date的目的在於後續計算市值加權時需要使用
            # 一開始寫的時候只儲存actual_date_entry，但加權之計算應該要由rebalance考量日(date_entry)為主
            # 因此在原先方法下會需要再回頭找一次t-1天，也就是實際交易日的前一天
            # 在交易筆數很多時會很慢，因此透過儲存date_entry可以在後續省去此步驟
            # 可以將更快依據date_entry將對應市值mapping回去各期欲交易之股票
            trade_table[(measure_date, actual_date_entry, actual_date_exit)] = temp_trade_table
            bar.update(i)
        bar.finish()  
        self.trade_table = trade_table
        if len(self.trade_table) == 0:
            raise Exception('無符合條件之股票，無法進行回測! /n 請重新設定條件')