from preprocess import Data
import pandas as pd
import numpy as np
from itertools import compress
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
from tabulate import tabulate
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA
# 避免在繪圖時跳出warning
# 參考：https://github.com/facebook/prophet/issues/999
pd.plotting.register_matplotlib_converters()

# 計算投資組合權重時需要使用市值
market_size = Data('market_size', data_frequency='day')
# 與大盤比較績效時需要使用benchmark
benchmark = Data('benchmark', data_frequency='day')
plt.style.use('seaborn')

class Analysis:
    def __init__(
        self,
        backtesting_result,
        tax_rate = 0.003,
        fee_rate = 0.001425
    ):
        '''
        針對回測結果backtesting_result進行績效分析。

        Parameters
        ----------
        backtesting_result:
            輸入回測結果
        tax_rate:
            交易稅率
        fee_rate: 
            手續費率
        
        Note
        ----
        1. 目前做多與放空在手續費的考量上相同，意謂若需要放空則沒有考量借(融)券成本
        '''
        self.tax_rate = tax_rate
        self.fee_rate = fee_rate
        # 由backtesting_result延用一些屬性
        self.trade_table = backtesting_result.trade_table
        self.qualified_market_size_signal = backtesting_result.qualified_market_size_signal
        self.entry_point = backtesting_result.entry_point
        self.start_date = backtesting_result.start_date
        self.end_date = backtesting_result.end_date
        # 讀入report_mode屬性，下方衡量進出場時還會用到(find_actual_entry_exit_date function)
        self.report_mode = backtesting_result.report_mode
        # 讀入long_or_short屬性，下方計算績效時會用到
        self.long_or_short = backtesting_result.long_or_short

        # 若完全沒有符合條件之交易，則停止以下物件之建立
        # 因為完全沒有交易則無法運行剩餘的程式
        total_signal_number = sum(list(map(
            lambda x: len(backtesting_result.entry_point[x]),
            backtesting_result.entry_point.keys()
        )))
        assert total_signal_number > 0, '無符合條件之交易，請重新設定策略條件！'

        # 可以發現market_size, benchmark沒有寫入class本身而trading_price有
        # 因為trading_price是由Backtest class所讀入，而此trading_price已經由
        # Backtest class中之回測區間縮減，因此接續讀入Analysis class
        # 可確保使用之回測時間皆相同，因此使用之資料範圍也相同
        # 雖然邏輯上來說即使另外讀取trading_price且不寫入Analysis class也沒差
        # 因為Analysis class中處理之訊號也來自Backtest class
        # 而此訊號也是經由Backtest class中縮減回測區間過後所產生
        # 基本上效果一樣，但透過Backtest class獲得trading_price更直覺一些
        self.trading_price = backtesting_result.trading_price
        self.long_or_short = backtesting_result.long_or_short

    def run(self):
        '''
        開始進行分析
        '''
        # 由於下方整理交易表格所需時間較長，因此使用progressbar顯示進度，以下設定相關outlay
        widgets = [
            'Progress: ', Percentage(), ' ',
            Bar('#'),' ', Timer(),
            ' , ', ETA(), ' '
        ]
        bar = ProgressBar(widgets=widgets, maxval=len(self.trade_table.keys())).start()
        # 整理交易表格
        # 將所有交易資訊存入self.trade_result中，持倉期間所有價格變化皆儲存
        # for 所有的進出場區間
        trade_result = pd.DataFrame()
        for i in range(len(self.trade_table.keys())):
            dates = list(self.trade_table.keys())[i]
            measure_date = dates[0]
            date_entry = dates[1]
            date_exit = dates[2]
            
            current_trade_table = self.trade_table[dates]
            # 用來裝同一期下各公司的結果
            temp_trade_table = pd.DataFrame()
            # current_trade_table.keys()是一個時間週期裡所有trading的公司
            for code in current_trade_table.keys():
                # 排除有同一天進出場的狀況，通常存在於最後一期(因為設定回測期間所導致的關係)
                if date_entry != date_exit:
                    # 用來裝單一公司的結果，之後會併入temp_trade_table
                    inner_temp_trade_table = pd.DataFrame([])
                    # 儲存資訊
                    # holding_date 如果是週交易會 keys() 會是一整周的交易日日期
                    # trading_price 如果是週交易 values() 會是一整周的交易價格
                    inner_temp_trade_table['holding_date'] = current_trade_table[code].keys()
                    inner_temp_trade_table['trading_price'] = current_trade_table[code].values()
                    inner_temp_trade_table['code'] = code
                    # 建立是否為進出場日之標記
                    inner_temp_trade_table['date_entry_or_exit'] = 0
                    
                    # 如果是週交易則entrydate 會標註1 中間的日期都是0 最後出場日會是-1
                    inner_temp_trade_table.loc[
                        inner_temp_trade_table.holding_date == date_entry, 'date_entry_or_exit'
                    ] = 1
                    inner_temp_trade_table.loc[
                        inner_temp_trade_table.holding_date == date_exit, 'date_entry_or_exit'
                    ] = -1
                    # 將判斷日期寫入同一期間之交易(之後可以當成key來運用)
                    inner_temp_trade_table['measure_date'] = measure_date
                    # 依據long_or_short計算報酬(做多或放空)
                    if self.long_or_short == 'long':
                        inner_temp_trade_table['return'] = inner_temp_trade_table.trading_price.pct_change()
                    elif self.long_or_short == 'short':
                        inner_temp_trade_table['return'] = -inner_temp_trade_table.trading_price.pct_change()
                    # 合併結果
                    temp_trade_table = pd.concat([
                        temp_trade_table, inner_temp_trade_table
                    ])

            # 將同一期下所有公司交易結果合併回self.trade_result
            trade_result = pd.concat([
                trade_result, temp_trade_table
            ]) 
            bar.update(i)
        bar.finish()  
        # 將持有第一期所導致沒有報酬的nan值填入0
        trade_result['return'].fillna(0, inplace=True)
        self.trade_result = trade_result

        # 計算兩種加權方式之個股權重
        # 首先計算1/N(naive)之加權方式
        # 計算各交易時期中投資組合公司家數
        # 因為給定holding_date與measure_date下會對應到相同時期評估下的交易標的
        # 若僅用holding_date則會出現同樣holding_date下對應到不同measure_date的狀況
        # 如此一來計算naive weight的權重會有問題
        # 因為在同holding_date下可能一批標的是上一期準備出場者，另一批則為下一批準備進場者
        # 所以同時依據holding_date和measure_date進行groupby可以確保各期中之交易公司家數正確
        number_of_stock = self.trade_result.groupby(['measure_date', 'holding_date']).count()['code']
        number_of_stock = number_of_stock.reset_index()
        number_of_stock.rename(columns={'code': 'number_of_stock'}, inplace=True)
        self.trade_result = self.trade_result.merge(number_of_stock, on=['holding_date', 'measure_date'])
        self.trade_result['naive_weight'] = 1/self.trade_result['number_of_stock']

        # 接著計算市值(cap_weight)加權方式
        # 定義functuion用於配對不同時期(measure_date)下各公司之市值
        def match_market_size(data):
            temp_data = self.qualified_market_size_signal[data.measure_date]
            return temp_data[data.code]

        # 依據measure_date去配對市值
        market_size_data = self.trade_result[self.trade_result.date_entry_or_exit == 1][['measure_date', 'code']]
        market_size_data['market_size'] = market_size_data.apply(
            lambda x: match_market_size(x), axis=1
        )

        # 將各期各公司之市值合併回去，同一measure_date下會有不同持有日
        # 但在同一measure_date下市值都會一樣
        self.trade_result = self.trade_result.merge(market_size_data, on=['code', 'measure_date'])

        # 將上述給定市值依據各期進行加總，並合併回去，所以各期會有單一個總市值
        total_market_size = market_size_data.groupby('measure_date')['market_size'].sum()
        total_market_size =  total_market_size.reset_index()
        total_market_size.rename(columns = {'market_size': 'total_market_size'}, inplace=True)
        self.trade_result = self.trade_result.merge(total_market_size, on='measure_date')
        self.trade_result['cap_weight'] = self.trade_result.market_size / self.trade_result.total_market_size

        # 計算投資組合報酬率
        # 由於需要考量成本，因此需要先計算持有部位變動表
        # 接著透過Cmoney回測中計算交易成本的方式進行交易成本計算
        # 公式為周轉率乘以來回交易成本(手續費*2 + 交易稅)
        # 周轉率計算方式：(本期買入數量+本期賣出數量)/2 / (上期持股數量+本期持股數量)
        # 以上都將以率的方式由毛報酬扣除
        # 首先計算持有變動表
        position_change_record = {}
        for i in range(len(self.entry_point.keys())):
            temp_position_change_record = {}
            last_date = list(self.entry_point.keys())[i]
            try:
                current_date = list(self.entry_point.keys())[i+1]
                date_entry = list(self.trade_table.keys())[i+1][1]
                date_exit = list(self.trade_table.keys())[i+1][2]
            except:
                break
                # current_date = list(self.entry_point.keys())[-1]
                # date_entry = list(self.trade_table.keys())[-1][1]
                # date_exit = list(self.trade_table.keys())[-1][2]
            # 上期持股與持股數量
            last_stock_pool = self.entry_point[last_date]
            num_of_stock_holding_last_period = len(last_stock_pool)
            # 本期持有股數與數量
            current_stock_pool = self.entry_point[current_date]
            num_of_stock_holding_current_period = len(current_stock_pool)
            # 本期作多持股與數量
            stock_bought_this_period = current_stock_pool.difference(last_stock_pool)
            num_of_stock_bought_this_period = len(stock_bought_this_period)
            # 本期平藏持股與數量
            stock_sold_this_period = last_stock_pool.difference(current_stock_pool)
            num_of_stock_sold_this_period = len(stock_sold_this_period)
            
            temp_position_change_record[
                'date_entry'
            ] = date_entry
            temp_position_change_record[
                'date_exit'
            ] = date_exit
            temp_position_change_record[
                'last_stock_pool'
            ] = list(last_stock_pool)
            temp_position_change_record[
                'num_of_stock_holding_last_period'
            ] = num_of_stock_holding_last_period
            temp_position_change_record[
                'current_stock_pool'
            ] = list(current_stock_pool)
            temp_position_change_record[
                'num_of_stock_holding_current_period'
            ] = num_of_stock_holding_current_period
            temp_position_change_record[
                'stock_bought_this_period'
            ] = list(stock_bought_this_period)
            temp_position_change_record[
                'num_of_stock_bought_this_period'
            ] = num_of_stock_bought_this_period
            temp_position_change_record[
                'stock_sold_this_period'
            ] = list(stock_sold_this_period)
            temp_position_change_record[
                'num_of_stock_sold_this_period'
            ] = num_of_stock_sold_this_period
            
            position_change_record[current_date] = temp_position_change_record
        # 將結果轉成Dataframe
        position_change_record = pd.DataFrame(position_change_record).T
        # 依據指定順序排
        position_change_record = position_change_record[[
            'date_entry', 'date_exit',
            'last_stock_pool', 'num_of_stock_holding_last_period',
            'current_stock_pool', 'num_of_stock_holding_current_period',
            'stock_bought_this_period', 'num_of_stock_bought_this_period',
            'stock_sold_this_period', 'num_of_stock_sold_this_period'
        ]]

        # 設置close_to_zero代替0避免division by zero的問題
        close_to_zero = 0.000000001
        position_change_record = position_change_record.replace(0, close_to_zero)

        # # 最前面兩期若沒有交易，在最後執行summary時會出現問題
        # # 主要原因是在計算賺賠比、勝率時，會使用到benchmark但因為資料
        # if len(position_change_record.current_stock_pool.iloc[0]) == 0:
        #     position_change_record = position_change_record.iloc[1:, ]
        # 計算周轉率，公式：(本期買入數量+本期賣出數量) / (上期持股數量+本期持股數量)
        position_change_record['turnover_rate'] = (
            ((
                position_change_record.num_of_stock_bought_this_period +
                position_change_record.num_of_stock_sold_this_period
            )) / (
                position_change_record.num_of_stock_holding_current_period +
                position_change_record.num_of_stock_holding_last_period
            )
        )

        # 由於上面有close_to_zero的設置，因此即使沒有交易還是有可能讓turnover_rate不為0
        # 所以要找出沒有進出場的點位，並設其為0
        index = (
            (position_change_record.num_of_stock_bought_this_period == close_to_zero) &
            (position_change_record.num_of_stock_sold_this_period == close_to_zero)
        )

        position_change_record.loc[index, 'turnover_rate'] = 0
        # 找出連續持有之股票˙
        position_change_record['cross_period_unchange_holding_stock'] = position_change_record[
            ['last_stock_pool', 'current_stock_pool']
        ].apply(
            lambda x: list(set(x['last_stock_pool']).intersection(
                set(x['current_stock_pool'])
            )),
            axis=1
        )
        self.position_change_record = position_change_record
        position_change_record.to_csv('position_change_record.csv')
        # 將turnover_rate單獨取出來並給予日期index
        # [index !=True]的目的在於濾掉前後期皆沒有交易的日期
        # 因為在下面要進行串表的時候，由於是採用outer方式所以會被考量進去
        # 雖然已透過上面步驟將這類交易日的turnover_rate轉為0
        # 但下面會有一個步驟是將第一期的周轉率設為0.5()，所以第一期出現這類狀況其週轉率便會被設成0.5
        # 如此一來便會有錯誤，因為在這種狀況下不應該有周轉率，應為0
        # 因此在產生turnover_rate時便先將前後其皆沒交易者濾掉
        # 可能會想為何不直接在源頭position_change_record便把這類交易drop掉
        # 但為了保有所有可交易期間之持股概況，即使沒有交易也應該保留其狀態，才能維持完整性
        self.turnover_rate = position_change_record['turnover_rate'][index !=True].reset_index()
        self.turnover_rate.rename(columns={'index': 'measure_date'}, inplace=True)
        self.turnover_rate['date_entry_or_exit'] = 1

        # 找出unique的進場日，後面將會作為key來串許多張表
        # 因為result中含有各檔股票的所有交易，因此會有重複的狀況
        # 透過groupby的方式可以找出unique的進場日
        key_for_measure_date = self.trade_result[[
            'holding_date', 'measure_date', 'date_entry_or_exit'
        ]]
        key_for_measure_date = key_for_measure_date[
            (key_for_measure_date.date_entry_or_exit == 1) |
            (key_for_measure_date.date_entry_or_exit == -1) 
        ]

        key_for_measure_date = key_for_measure_date.groupby(
            ['holding_date', 'measure_date', 'date_entry_or_exit']
        ).count()
        key_for_measure_date = key_for_measure_date.reset_index()

        # 定義計算drawdown的function，在下面turnover_rate會使用到
        # 會獨立寫成function是因為計算benchmark時也需要用到
        # 而計算benchmark將會獨立計算，如此一來code會重複，因此獨立寫成function
        def calculate_drawdown(return_data):
            drawdown_list = []
            for i in range(len(return_data.cumulative_return)):
                current_cumulative_return = return_data.cumulative_return.iloc[i]
                if i == 0:
                    new_highest = return_data.cumulative_return.iloc[i]
                if (current_cumulative_return > new_highest) and (current_cumulative_return > 0):
                    new_highest = current_cumulative_return

                drawdown = (
                    (current_cumulative_return - new_highest) / new_highest
                )
                if drawdown < 0:
                    drawdown_list.append(drawdown)
                else:
                    drawdown_list.append(0)
            return drawdown_list

        # 計算turnover_rate後接著計算淨報酬，步驟較繁瑣因此用function包起來
        def calculate_net_return(weight_type):
            weight_return = self.trade_result.groupby(
                ['holding_date', 'measure_date']
            ).sum()[weight_type]

            weight_return = weight_return.reset_index()
            
            # 讓weight_return產生date_entry_or_exit的欄位
            weight_return = weight_return.merge(
                key_for_measure_date,
                on=['holding_date', 'measure_date'],
                how='outer'
            )

            weight_return = weight_return.merge(
                self.turnover_rate,
                on=['measure_date', 'date_entry_or_exit'],
                how='outer'
            )
            
            # 接著要找出當期沒有交易但有出場者，並填入holding_date資訊
            # 否則會少記到當期出場之成本(雖然沒有交易，但仍可能會有出場)
            # 由於trade_result中若當期沒新買入之個股就不會有交易紀錄(但仍可能有賣出)
            # 定義function找尋距離公布日最近的當個或次個交易日
            def find_actual_entry_exit_date(date):
                # 透過report_mode判斷欲使用當個或次個價格進出場
                if self.report_mode == False:
                    temp_index = self.trading_price.data.columns > date
                elif self.report_mode == True:
                    temp_index = self.trading_price.data.columns >= date

                try:
                    temp_target = self.trading_price.data.columns[temp_index]
                    target = min(temp_target)
                except:
                    target = self.trading_price.data.columns[-1]
                return target
            
            # 一開始串資料表時，由於trade_result本身不會記錄沒有做多的期數
            # 所以串資料表時會出現na值，透過相對應之na值位置便可以找到缺值處
            # 接著依據measure_date找出最近的次一個交易日，並填回指定na位置
            weight_return.loc[
                weight_return.holding_date.isna(), 'holding_date'
            ] = weight_return[
                weight_return.holding_date.isna()
            ].apply(
                lambda x: find_actual_entry_exit_date(x['measure_date']), axis=1
            )
            # 此種情況下不會有return，因此將return指定為0
            weight_return[weight_type].fillna(0, inplace=True)
            # 由於計算累積報酬時需要考量順序，因此依據日期排序
            weight_return.sort_values('holding_date', inplace=True)
            # 重新排序index
            weight_return.reset_index(drop=True, inplace=True)
            
            # 由於turnover_rate是從position_change_record而來
            # 但position_change_record中的第一期並不會被記錄turnover_rate
            # position_change_record中會考量到上一期持股數量，因此會從第二期開始
            # 第一期即使沒有上一期資訊，但仍會有當期進場的成本需要考量
            # 而第一期的turnover_rate為0.5，因此以下直接指定其為0.5
            # 因為第一期的turnover rate在position_change_record中並不會被考量，所以這裡直接指定
            weight_return.loc[0, 'turnover_rate'] = 0.5

            weight_return['transaction_cost_rate'] = weight_return.turnover_rate * (
                self.fee_rate*2 + self.tax_rate
            )

            weight_return.transaction_cost_rate.fillna(0, inplace=True)
            
            weight_return['net_{}'.format(weight_type)] = (
                weight_return[weight_type] - weight_return.transaction_cost_rate
            )

            # 建倉當日持有股票不會有報酬(報酬衡量方式由今日開盤與上一日開盤計算)
            # 但會有上一期建倉重新平衡前最後一筆之報酬，對應權重也會是上一期計算之權重
            # 因此會出現兩個相同holding_date的狀況，但分別來自不同的measure_date
            # 可由date_entry_or_exit來搭配判斷為再平衡時的哪一操作
            # 若date_entry_or_exit為-1則為上一期之持股結倉
            # 若date_entry_or_exit為1則為下一期之持股開倉
            # 換倉手續費則算入date_entry_or_exit為1之開倉日
            # 但以下在算績效指標時會用到benchmark的報酬序列與策略報酬序列
            # 但由於策略報酬序列會有同一天出現兩個交易日的狀況，因此要縮減到同一天內
            # 以下做法是在同一持有日下，依據date_entry_or_exit把結倉日排在前面，換倉日則排在後面
            # 接著把報酬加1後，累乘起來後取後面的累乘結果(第二筆)，接著把每期第一筆多於者剃除
            # 最後在計算累積報酬
            
            # 依據持有日排接著由date_entry_or_exit排，-1排至1
            weight_return.sort_values(
                ['holding_date', 'date_entry_or_exit'], inplace=True
            )

            # 加1後接著groupby累乘，主要目的是要把重複者(同持有日有兩筆)累乘起來
            weight_return['net_{}'.format(weight_type)] = weight_return[
                'net_{}'.format(weight_type)
            ] + 1

            weight_return['net_{}'.format(weight_type)] = weight_return.groupby('holding_date')[
                'net_{}'.format(weight_type)
            ].cumprod()
            
            # 只保留後面累乘結果
            weight_return = weight_return[
                weight_return.duplicated('holding_date', keep='last') != True
            ]

            # 算完後把報酬減1
            weight_return['net_{}'.format(weight_type)] = weight_return[
                'net_{}'.format(weight_type)
            ] - 1
            
            # 計算累積報酬
            weight_return['cumulative_return'] = np.cumprod(
                weight_return['net_{}'.format(weight_type)] + 1
            )
            
            # 計算drwadown前先reset_index
            weight_return.reset_index(inplace=True, drop=True)
            weight_return['drawdown'] = calculate_drawdown(weight_return)
            
            return weight_return

        self.trade_result['naive_weight_return'] = (
            self.trade_result.naive_weight * self.trade_result['return']
        )
        self.trade_result['cap_weight_return'] = (
            self.trade_result.cap_weight * self.trade_result['return']
        )
        self.naive_weight_return = calculate_net_return('naive_weight_return')
        self.cap_weight_return = calculate_net_return('cap_weight_return')

        # 計算benchmark報酬
        benchmark_data = benchmark.data.T
        benchmark_data = benchmark_data[
            (benchmark_data.index >= self.start_date) &
            (benchmark_data.index <= self.end_date)
        ]
        benchmark_data['holding_date'] = benchmark_data.index
        
        # 計算報酬率
        benchmark_data['return'] = benchmark_data['TWA02'].pct_change()
        # 第一期為na，因此填入0
        benchmark_data['return'].fillna(0, inplace=True)
        
        # 計算大盤報酬時不會考量股票是否有部位
        benchmark_data['cumulative_return'] = np.cumprod(
            benchmark_data['return'] + 1
        )

        # 計算benchmark
        benchmark_data['drawdown'] = calculate_drawdown(benchmark_data)
        self.benchmark_data = benchmark_data
    
        # 計算benchmark 與 兩種 weighting return 的差異
        # 由於benchmark_data的長度可能會比策略產出之績效還長
        # 因此需要在這裏對資料進行縮減，才能找出同日期下的difference
        benchmark_data_for_difference = self.benchmark_data
        benchmark_data_for_difference = benchmark_data_for_difference[
            benchmark_data_for_difference.index.isin(
                self.naive_weight_return.holding_date
            )
        ]
        self.benchmark_data_for_difference = benchmark_data_for_difference

        # 第一期為na，因此填入0

        
        
    # 定義計算報酬的function(給定頻率下)
    def __calculate_return_for_given_frequency(
        self,
        weight_method='naive_weight',
        frequency='month',
        compare_benchmark='N'
    ):
        if weight_method == 'naive_weight':
            temp_weight_return = self.naive_weight_return
        else:
            temp_weight_return = self.cap_weight_return
            
            
        # benchmark_return =
        # 建立日期key，稍後計算各時間頻率之報酬時會用到
        temp_weight_return['year'] = temp_weight_return.holding_date.apply(
            lambda x: x.year
        )

        temp_weight_return['quarter'] = temp_weight_return.holding_date.apply(
            lambda x: x.quarter
        )

        temp_weight_return['month'] = temp_weight_return.holding_date.apply(
            lambda x: x.month
        )
        
        # 計算給定頻率下之報酬
        temp_return_by_period = temp_weight_return.groupby(['year', frequency]).apply(
            lambda x: x.cumulative_return.iloc[-1] / x.cumulative_return.iloc[0] - 1
        )
        # 如果要跟Benchmark比較的話
        if compare_benchmark == 'Y':
            self.benchmark_data_for_difference.loc[:,'year'] = temp_weight_return['year'].values
            self.benchmark_data_for_difference.loc[:,'quarter'] = temp_weight_return['quarter'].values
            self.benchmark_data_for_difference.loc[:,'month'] = temp_weight_return['month'].values
            
            temp_return_by_period_bench = self.benchmark_data_for_difference.groupby(
                ['year', frequency]).apply(
                lambda x: x.cumulative_return.iloc[-1] / x.cumulative_return.iloc[0] - 1
            )
            temp_return_by_period = temp_return_by_period - temp_return_by_period_bench

        # 將計算後之報酬整理成欲繪圖的格式
        # 原先會是MultiIndex的series，要轉成pd.Dataframe
        if frequency != 'year':
            temp_return_by_period = temp_return_by_period.reset_index()
            temp_return_by_period.rename(
                columns={0: weight_method},
                inplace=True
            )
        # 如果頻率是年需要另外處理，因為year本身就在key中
        # 在reset_index時會出現名稱重複無法插入的狀況，所以處理步驟較繁複
        else:
            year_index = list(map(
                # 取出MultiIndex中的第一個數值(其實第二個也可以，ex. (2009, 2009))
                lambda x: x[0],
                temp_return_by_period.index.values
            ))
            temp_return_by_period = pd.DataFrame(temp_return_by_period.values)
            temp_return_by_period['year'] = year_index
            temp_return_by_period.rename(
                columns={0: weight_method},
                inplace=True
            )
        
        return temp_return_by_period

    def plot_cumulative_return(
        self,
        weight_method='naive_weight',
        benchmark=True,
        below_x_axis = 'underperformance', 
        figsize=(20, 10),
        title_fontsize=25,
        xlabel_fontsize=20,
        ylabel_fontsize=20,
        xticks_fontsize=15,
        yticks_fontsize=15,
        legend_fontsize=16
    ):
        '''
        繪製策略累積報酬圖，同時可以顯示不同加權方式與大盤報酬。

        Parameters
        ----------
        weight_method:
            選擇投資組合加權方式：
            - naive_weight: 1/N加權方式
            - cap_weight: 市值加權
            - both: 同時繪製兩種加權方式
        benchmark:
            是否開啟benchmark進行比較：
            - True: 開啟
            - False: 關閉
        below_x_axis:
            y軸以下繪製之資訊：
            - underperformance: 策略落後於大盤之績效
            - drawdown: 策略本身的drawdown
        figsize：
            調整圖表大小(int, float)，格式(長, 寬)
        title_fontsize:
            調整標題大小(float)
        xticks_fontsize:
            調整x軸座標大小(float)
        yticks_fontsize:
            調整y軸座標大小(float)
        xlabel_fontsize:
            調整x軸座標名稱大小(float)
        ylabel_fontsize:
            調整y軸座標名稱大小(float)
        legend_fontsize:
            調整字體大小(float)

        Notes
        -----
        1. 由於使用回測之股價皆有還原權息，因此採用台灣發行量加權股價報酬指數(TWA02)作為benchmark
        '''
        plt.figure(figsize=figsize)

        # 建立function繪圖，可以彈性選擇要畫的權重方式，同時避免重複的code
        def plot_equity_curve(
            weight_method,
            cumulative_return_color='black',
            alpha=1
        ):
            # 由於benchmark_data的長度可能會比策略產出之績效還長
            # 因此需要在這裏對資料進行縮減，才能找出同日期下的difference
            benchmark_data_for_difference = self.benchmark_data
            benchmark_data_for_difference = benchmark_data_for_difference[
                benchmark_data_for_difference.index.isin(
                    self.naive_weight_return.holding_date
                )
            ]

            
            # print('naive_weight_return',self.naive_weight_return)
            # print('benchmark_data_', benchmark_data_for_difference)
            if weight_method == 'naive_weight':
                cumulative_return = self.naive_weight_return.cumulative_return
                drawdown = list(self.naive_weight_return.drawdown)
                # 落後於大盤之績效，可自行選擇是否繪製
                difference = pd.DataFrame([
                    self.naive_weight_return.cumulative_return.values -
                    benchmark_data_for_difference.cumulative_return.values
                ]).T
            else:
                cumulative_return = self.cap_weight_return.cumulative_return
                drawdown = list(self.cap_weight_return.drawdown)
                # 落後於大盤之績效，可自行選擇是否繪製 
                difference = pd.DataFrame([
                    self.cap_weight_return.cumulative_return.values -
                    benchmark_data_for_difference.cumulative_return.values
                ]).T

            cumulative_return.index = self.naive_weight_return.holding_date
            difference.columns = ['difference']
            difference.index = benchmark_data_for_difference.index
            difference.loc[difference.difference > 0, 'difference'] = 0
            
            new_highest_index = []
            for i in range(len(cumulative_return)):
                current_cumulative_return = cumulative_return.iloc[i]
                if i == 0:
                    new_highest = cumulative_return.iloc[i]
                if (current_cumulative_return > new_highest) and (current_cumulative_return > 0):
                    new_highest = current_cumulative_return
                    new_highest_index.append(i)
            
            # 畫出累積權益曲線
            plt.plot(
                cumulative_return,
                c=cumulative_return_color,
                label='Cumulative return (100%) - {}'.format(weight_method)
            )
            # 畫出0軸
            plt.plot(
                pd.to_datetime(self.naive_weight_return.holding_date),
                [0]*len(self.naive_weight_return.holding_date),
                c='grey', alpha=alpha
            )
            # 畫出drawdown，並將其與0之間的距離填滿紅色
            if below_x_axis == 'drawdown':
                plt.fill_between(
                    pd.to_datetime(self.naive_weight_return.holding_date),
                    drawdown, 0,
                    facecolor='red',
                    alpha=alpha,
                    label='Drawdown - {}'.format(weight_method)
                )
            elif below_x_axis == 'underperformance':
                plt.fill_between(
                    pd.to_datetime(difference.index),
                    difference.difference.values, 0,
                    facecolor='red',
                    alpha=alpha,
                    label='Underperformance - {}'.format(weight_method)
                )

            # 畫出創高點
            plt.scatter(
                cumulative_return.iloc[new_highest_index].index,
                cumulative_return.iloc[new_highest_index].values,
                s=70, c='#02ff0f', alpha=alpha,
                label='Historical high - {}'.format(weight_method)
            )
        
        if weight_method == 'both':
            plot_equity_curve('naive_weight')
            plot_equity_curve(
                'cap_weight',
                cumulative_return_color='grey',
                alpha=0
            )
        else:
            plot_equity_curve(weight_method)

        # 畫出Benchmark
        if benchmark == True:
            plt.plot(
                self.benchmark_data.cumulative_return,
                label='Benchmark (100%) - TWA02',
                color='#1959d1'
            )
        
        # 調整x, y軸之ticks字體大小
        plt.tick_params(axis="x", labelsize=xticks_fontsize)
        plt.tick_params(axis="y", labelsize=yticks_fontsize)
        
        plt.title('Cumulative Return', fontsize=title_fontsize)
        plt.xlabel('Year', fontsize=xlabel_fontsize)
        plt.ylabel('Cumulative Return', fontsize=ylabel_fontsize)

        plt.legend(fontsize=legend_fontsize)
        plt.autoscale(axis='both');

    def plot_profit_and_loss(
        self,
        weight_method='naive_weight',
        frequency='month',
        figsize=(20, 10),
        title_fontsize=25,
        xlabel_fontsize=20,
        ylabel_fontsize=20,
        xticks_fontsize=15,
        yticks_fontsize=15,
        legend_fontsize=16,
        spacing=5,
        xticks_rotation_angle=45,
    ):
        '''
        繪製給定頻率下的績效長條圖。

        Parameters
        ----------
        weight_method:
            選擇投資組合加權方式：
            - naive_weight: 1/N加權方式
            - cap_weight: 市值加權
        frequency:
            欲用來衡量績效的時間頻率：
            - month: 月
            - quarter: 季
            - year: 年
        figsize：
            調整圖表大小(int, float)，格式(長, 寬)
        title_fontsize:
            調整標題大小(float)
        xticks_fontsize:
            調整x軸座標大小(float)
        yticks_fontsize:
            調整y軸座標大小(float)
        xlabel_fontsize:
            調整x軸座標名稱大小(float)
        ylabel_fontsize:
            調整y軸座標名稱大小(float)
        legend_fontsize:
            調整字體大小(float)
        spacing:
            調整x軸座標間隔數量，數字越小則x軸座標越多，spacing必須大於0(最小為1)。
        xticks_rotation_angle:
            將x軸座標旋轉角度
        '''
        # 定義一些function下面繪圖將被使用
        # 找出序列值為正或負的index(區分顏色差異)
        def get_index(series, greater_smaller_than_0='greater'):
            temp_sereis = series.reset_index()
            if greater_smaller_than_0 == 'greater':
                return temp_sereis.iloc[:, 1][temp_sereis.iloc[:, 1] >= 0].index
            else:
                return temp_sereis.iloc[:, 1][temp_sereis.iloc[:, 1] < 0].index
        
        # 使用自定義的function計算報酬
        return_by_period = pd.merge(
            self.__calculate_return_for_given_frequency(
                'naive_weight', frequency=frequency
            ),
            self.__calculate_return_for_given_frequency(
                'cap_weight', frequency=frequency
            ),
            on=['year', frequency]
        )

        # 設定畫布大小
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        # 設定x軸到時將使用的刻度
        index = np.arange(len(return_by_period))
        if frequency == 'quarter':
            date_xticks = return_by_period.apply(
                lambda x: datetime(int(x.year), int(x.quarter)*3, 1),
                axis=1
            )
            date_xticks = pd.PeriodIndex(date_xticks, freq='Q')
        elif frequency == 'month':
            date_xticks = return_by_period.apply(
                lambda x: datetime(int(x.year), int(x.month), 1),
                axis=1
            )
            date_xticks = pd.PeriodIndex(date_xticks, freq='m')
        elif frequency == 'year':
            date_xticks = return_by_period.year

        # 繪製長條圖
        ax.bar(
            get_index(return_by_period[weight_method], 'greater'),
            return_by_period[weight_method][get_index(return_by_period[weight_method])],
            color='red', alpha=1, align='center', label='Profit'
        )

        ax.bar(
            get_index(return_by_period[weight_method], 'smaller'),
            return_by_period[weight_method][get_index(return_by_period[weight_method], 'smaller')],
            color='green', alpha=1, align='center', label='Loss'
        )

        # 畫出次要加權方式之損益(散佈圖)
        if weight_method == 'naive_weight':
            ax.scatter(
                index, return_by_period['cap_weight'],
                marker='x', label='Profit and Loss ({})'.format('cap weight'),
                c='#002e78', zorder=10
            )
        else:
            ax.scatter(
                index, return_by_period['naive_weight'],
                marker='x', label='Profit and Loss ({})'.format('naive weight'),
                c='#002e78', zorder=10
            )

        # 將x軸的ticks轉成日期
        plt.xticks(
            range(len(return_by_period)), date_xticks,
            rotation=xticks_rotation_angle
        )
        visible = ax.xaxis.get_ticklabels()[::spacing]
        for label in ax.xaxis.get_ticklabels():
            if label not in visible:
                label.set_visible(False)

        # 調整x, y軸之ticks字體大小
        ax.tick_params(axis="x", labelsize=xticks_fontsize)
        ax.tick_params(axis="y", labelsize=yticks_fontsize)

        # 調整網格
        ax.grid(which='major', color='#CCCCCC', linestyle='--')
        ax.grid(which='minor', color='#CCCCCC', linestyle=':')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=1)

        # 加入title
        plt.title(
            'Return of Profit and Loss ({}) - by {}'.format(
                weight_method.replace('_', " "), frequency
            ),
            fontsize=title_fontsize
        )
        plt.xlabel('Date of trade exited', fontsize=xlabel_fontsize)
        plt.ylabel('Profit and Loss', fontsize=ylabel_fontsize)

        ax.legend(fontsize=legend_fontsize)
        ax.autoscale(axis='both');


    def plot_return_heatmap(
        self,
        weight_method='naive_weight',
        compare_benchmark='N',
        figsize=(12, 6),
        title_fontsize=20,
        xticks_fontsize=10,
        yticks_fontsize=10, 
        xlabel_fontsize=15,
        ylabel_fontsize=15
    ):
        '''
        繪製給定年月下之報酬熱點圖。

        Parameters
        ----------
        weight_method:
            選擇投資組合加權方式：
            - naive_weight: 1/N加權方式
            - cap_weight: 市值加權
        figsize：
            調整圖表大小(int, float)，格式(長, 寬)
        title_fontsize:
            調整標題大小(float)
        xticks_fontsize:
            調整x軸座標大小(float)
        yticks_fontsize:
            調整y軸座標大小(float)
        xlabel_fontsize:
            調整x軸座標名稱大小(float)
        ylabel_fontsize:
            調整y軸座標名稱大小(float)
        '''
        
        # 這個return 計算出來是 accumulative return
        self.month_return = self.__calculate_return_for_given_frequency(
            weight_method=weight_method, compare_benchmark=compare_benchmark
        )

        months = np.arange(1, 13, 1)
        years = np.sort(np.unique(self.month_return.year))

        # 建立繪製heatmap所需之資料，依據年月日填入資料
        data_for_heatmap = pd.DataFrame()
        for year in years:
            inner_result = []
            for month in months:
                temp_result = self.month_return[self.month_return.year == year]
                temp_result = temp_result[temp_result.month == month]
                try:
                    # 選擇欲使用之加權方式
                    if weight_method == "naive_weight":
                        weight_return = temp_result.naive_weight.values[0]
                    else:
                        weight_return = temp_result.cap_weight.values[0]
                except:
                    weight_return = 0
                inner_result.append(weight_return)
            outter_result = pd.DataFrame([inner_result])
            outter_result.columns = months
            outter_result.index = [year]
            data_for_heatmap = pd.concat([data_for_heatmap, outter_result])

        data_for_heatmap = data_for_heatmap.T
        data_for_heatmap = data_for_heatmap*100
        data_for_heatmap = data_for_heatmap.round(2)

        # 繪製heatmap
        plt.figure(figsize=figsize)

        top = cm.get_cmap('Greens_r', 64)
        bottom = cm.get_cmap('Reds', 64)

        newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                            bottom(np.linspace(0, 1, 128))))
        newcmp = ListedColormap(newcolors, name='OrangeBlue')

        ax = sns.heatmap(
            data_for_heatmap, annot=True, cmap=newcmp,
            vmax=data_for_heatmap.max().max(),
            vmin=-data_for_heatmap.max().max()
        )

        # 調整x, y軸之ticks字體大小
        ax.set_xticklabels(years, size=xticks_fontsize)
        ax.set_yticklabels(months, size=yticks_fontsize, rotation=360)

        plt.tight_layout()
        plt.title('Return of Profit and Loss (%)', size=title_fontsize)
        plt.xlabel('Year', size=xlabel_fontsize)  
        plt.ylabel('Month', size=ylabel_fontsize);

    def summary(self):
        '''
        呈現策略績效總表，三個columns分別包含以下資訊：
        大盤(benchmark)、naive_weight(等權重)、cap_weight(市值加權)的績效結果。
        '''
        # 將float轉換成百分比形式，同時轉成str，因為需要呈現%符號
        def transform_to_percentage(x):
            return str(round(100*x, 2)) + '%'
        
        # 策略起始日
        start_date = self.start_date
        start_date = start_date.strftime('%Y/%m/%d')
        
        # 策略結束日
        end_date = self.end_date
        end_date = end_date.strftime('%Y/%m/%d')
        
        # 策略運行年數
        trading_years = (self.end_date - self.start_date).days / 365
        trading_years = round(trading_years, 2)
        
        # 總交易筆數(單邊)
        total_trade_num = len(
            self.trade_result[self.trade_result.date_entry_or_exit == 1]
        )
        
        # 總交易期數(第一期建立部位時不會被記錄進position_change_record，因此要加一)
        total_trade_period = len(self.position_change_record) + 1
        
        # 平均每期交易次數
        trade_num_per_period = round(
            total_trade_num/total_trade_period, 2
        )
        
        # 平均每期持有天數
        holding_days_per_period = (
            pd.DataFrame(list(self.entry_point.keys())) -
            pd.DataFrame(list(self.entry_point.keys())).shift(1)
        ).mean()[0].days
        
        # 總累積報酬率(%)
        cumulative_return_benchmark = transform_to_percentage(
            (self.benchmark_data.cumulative_return.iloc[-1] - 1)
        )
        
        cumulative_return_naive_weight = transform_to_percentage(
            (self.naive_weight_return.cumulative_return.iloc[-1] - 1)
        )
        
        cumulative_return_cap_weight = transform_to_percentage(
            (self.cap_weight_return.cumulative_return.iloc[-1] - 1)
        )

        # 計算年化報酬
        annualize_factor = 250
        annual_return_benchmark = self.benchmark_data[
            'return'
        ].mean() * annualize_factor
        annual_return_benchmark = transform_to_percentage(annual_return_benchmark)

        annual_return_naive_weight = self.naive_weight_return[
            'net_naive_weight_return'
        ].mean() * annualize_factor
        annual_return_naive_weight = transform_to_percentage(annual_return_naive_weight)

        annual_return_cap_weight = self.cap_weight_return[
            'net_cap_weight_return'
        ].mean() * annualize_factor
        annual_return_cap_weight = transform_to_percentage(annual_return_cap_weight)
        
        # CAGR
        CAGR_benchmark = np.power(
            (self.benchmark_data.cumulative_return.iloc[-1]), (1/trading_years)
        ) - 1
        CAGR_benchmark = transform_to_percentage(CAGR_benchmark)
        
        CAGR_naive_weight = np.power(
            (self.naive_weight_return.cumulative_return.iloc[-1]), (1/trading_years)
        ) - 1
        CAGR_naive_weight = transform_to_percentage(CAGR_naive_weight)
        
        CAGR_cap_weight = np.power(
            (self.cap_weight_return.cumulative_return.iloc[-1]), (1/trading_years)
        ) - 1
        CAGR_cap_weight = transform_to_percentage(CAGR_cap_weight)
        
        # 最高累積權益(%)
        max_cumulative_return_benchmark = transform_to_percentage(
            (self.benchmark_data.cumulative_return.max() - 1)
        )
        
        max_cumulative_return_naive_weight  = transform_to_percentage(
            (self.naive_weight_return.cumulative_return.max() - 1)
        )
        
        max_cumulative_return_cap_weight  = transform_to_percentage(
            (self.cap_weight_return.cumulative_return.max() - 1)
        )
        
        # 最低累積權益(%)
        min_cumulative_return_benchmark  = transform_to_percentage(
            (self.benchmark_data.cumulative_return.min() - 1)
        )
        
        min_cumulative_return_naive_weight  = transform_to_percentage(
            (self.naive_weight_return.cumulative_return.min() - 1)
        )
        
        min_cumulative_return_cap_weight  = transform_to_percentage(
            (self.cap_weight_return.cumulative_return.min() - 1)
        )
        
        # Max drawdown
        max_drawdown_benchmark = transform_to_percentage(
            min(self.benchmark_data.drawdown)
        )
        
        max_drawdown_naive_weight = transform_to_percentage(
            min(self.naive_weight_return.drawdown)
        )
        
        max_drawdown_cap_weight = transform_to_percentage(
            min(self.cap_weight_return.drawdown)
        )

        # 風險報酬比(恢復能力，總報酬/max drawdown)
        total_return_over_max_drawdown_benchmark = (
            (self.benchmark_data.cumulative_return.iloc[-1] - 1) /
            abs(min(self.benchmark_data.drawdown))
        )
        total_return_over_max_drawdown_benchmark = round(
            total_return_over_max_drawdown_benchmark, 2
        )

        total_return_over_max_drawdown_naive_weight = (
            (self.naive_weight_return.cumulative_return.iloc[-1] - 1) /
            abs(min(self.naive_weight_return.drawdown))
        )
        total_return_over_max_drawdown_naive_weight = round(
            total_return_over_max_drawdown_naive_weight, 2
        )
        
        total_return_over_max_drawdown_cap_weight = (
            (self.cap_weight_return.cumulative_return.iloc[-1] - 1) /
            abs(min(self.cap_weight_return.drawdown))
        )
        total_return_over_max_drawdown_cap_weight = round(
            total_return_over_max_drawdown_cap_weight, 2
        )

        # 與大盤之相對報酬
        naive_weight_and_benchmark_pair = pd.merge(
            self.benchmark_data[[
                'holding_date', 'cumulative_return', 'return'
            ]],
            self.naive_weight_return[[
                'holding_date', 'cumulative_return', 'net_naive_weight_return'
            ]],
            on='holding_date'
        )

        max_outperformance_naive_weight = max(
            naive_weight_and_benchmark_pair.cumulative_return_y - 
            naive_weight_and_benchmark_pair.cumulative_return_x
        )
        max_outperformance_naive_weight = transform_to_percentage(
            max_outperformance_naive_weight
        )
        
        min_outperformance_naive_weight = min(
            naive_weight_and_benchmark_pair.cumulative_return_y - 
            naive_weight_and_benchmark_pair.cumulative_return_x
        )
        min_outperformance_naive_weight = transform_to_percentage(
            min_outperformance_naive_weight
        )
        
        cap_weight_and_benchmark_pair = pd.merge(
            self.benchmark_data[[
                'holding_date', 'cumulative_return', 'return'
            ]],
            self.cap_weight_return[[
                'holding_date', 'cumulative_return', 'net_cap_weight_return'
            ]],
            on='holding_date'
        )

        max_outperformance_cap_weight = max(
            cap_weight_and_benchmark_pair.cumulative_return_y - 
            cap_weight_and_benchmark_pair.cumulative_return_x
        )
        max_outperformance_cap_weight = transform_to_percentage(
            max_outperformance_cap_weight
        )
        
        min_outperformance_cap_weight = min(
            cap_weight_and_benchmark_pair.cumulative_return_y - 
            cap_weight_and_benchmark_pair.cumulative_return_x
        )
        min_outperformance_cap_weight = transform_to_percentage(
            min_outperformance_cap_weight
        )
        
        # 計算連續虧損天數之function
        def calculate_max_under_water_days(weight_method):
            counter = 0
            counter_list = []
            if weight_method == 'naive_weight':
                drawdown = self.naive_weight_return.drawdown
            elif weight_method == 'cap_weight':
                drawdown = self.cap_weight_return.drawdown 
            elif weight_method == 'benchmark':
                drawdown = self.benchmark_data.drawdown
            
            for i in range(len(drawdown)):
                current_drawdown = drawdown[i]
                if current_drawdown < 0:
                    counter += 1
                # 最後一期則終止計算
                if (current_drawdown == 0) | (i == len(drawdown)-1):
                    counter_list.append(counter)
                    counter = 0
                    
            return max(counter_list)
        
        # 最長連續虧損天數
        max_under_water_days_benchmark = calculate_max_under_water_days('benchmark')
        
        max_under_water_days_naive_weight = calculate_max_under_water_days('naive_weight')
        
        max_under_water_days_cap_weight = calculate_max_under_water_days('cap_weight')
        
        # 勝率
        winning_rate_benchmark = (
            sum(self.benchmark_data['return'] > 0)/
            len(self.benchmark_data['return'])
        )
        winning_rate_benchmark = transform_to_percentage(winning_rate_benchmark)
        
        
        winning_rate_naive_weight = (
            sum(self.naive_weight_return.net_naive_weight_return > 0)/
            len(self.naive_weight_return.net_naive_weight_return)
        )
        winning_rate_naive_weight = transform_to_percentage(winning_rate_naive_weight)
        
        
        
        winning_rate_cap_weight = (
            sum(self.cap_weight_return.net_cap_weight_return > 0)/
            len(self.cap_weight_return.net_cap_weight_return)
        )
        winning_rate_cap_weight = transform_to_percentage(winning_rate_cap_weight)
        
        # 以下計算賺賠比，計算賺賠比本身有兩種方式
        # 基本上在各交易期間之各股票都會有各自的損益，接下來依據不同方式計算賺賠比與勝率
        # 第一種是依據各交易時間週期計算各期間之賺賠比與勝率，接著將各期間的單一結果平均
        # 第二種則是不考量交易時間週期，而直接把所有結果跨週期平均(賺賠比)，勝率則是直接衡量所有交易
        # 上述兩種衡量方式外，比較門檻除了0之外也需與大盤相比
        # 因此上述動作需將比較門檻轉換為大盤報酬再比較一次
        # 因此共會有2 x 2，4種結果
        # 上述計算均與權重無關，單純看個股報酬狀況
        # 首先計算各間各股票之報酬
        stock_trading_price = self.trade_result[
            ['holding_date', 'trading_price', 'code', 'measure_date']
        ]

        def calculate_return(x):
            if self.long_or_short == 'long':
                return (x['trading_price'].iloc[-1] / x['trading_price'].iloc[0]) -1
            else:
                return -((x['trading_price'].iloc[-1] / x['trading_price'].iloc[0]) -1)

        result = stock_trading_price.groupby(['code', 'measure_date']).apply(
            lambda x: calculate_return(x)
        ).reset_index()

        result.rename(columns={0: 'return'}, inplace=True)

        # 依據策略之交易日期整理出同期間的大盤報酬
        # 依據日期整理出表格
        date_entry_of_benchmark = pd.DataFrame(list(map(
            # 萃取出各重要日期
            lambda x: [x[0], x[1], x[2]],
            list(self.trade_table.keys())
        )), columns=['measure_date', 'date_entry', 'date_exit'])

        # 依據日期填入對應的大盤價格
        for i in range(len(date_entry_of_benchmark)):
            date_entry = date_entry_of_benchmark['date_entry'][i]
            date_exit = date_entry_of_benchmark['date_exit'][i]
            # 由於date_entry_of_benchmark是由self.trade_table所組成
            # 而self.trade_table為所有可交易時間，並不會考量是否有實際交易
            # 因此依據date_entry_of_benchmark進行迴圈，會碰到沒有交易的日子
            # 但benchmark_data已經經過實際交易日之縮減，因此迴圈要向benchmark_data要資料時
            # 便會要不到無交易的日期，因為已從benchmark_data中被剔除
            # 如此一來會出現'index out of bound'的error，所以透過try去避免
            try:
                date_entry_of_benchmark.loc[i, 'entry_price'] = self.benchmark_data[
                    self.benchmark_data.index == date_entry
                ]['TWA02'][0]
                date_entry_of_benchmark.loc[i, 'exit_price'] = self.benchmark_data[
                    self.benchmark_data.index == date_exit
                ]['TWA02'][0]
            except: pass

        # 計算報酬
        date_entry_of_benchmark['benchmark_return'] = (
            date_entry_of_benchmark.exit_price / 
            date_entry_of_benchmark.entry_price
        ) - 1

        # 把計算後的大盤報酬合併回去
        result = result.merge(
            date_entry_of_benchmark[['measure_date', 'benchmark_return']],
            on='measure_date'
        )

        # 計算各交易期間報酬大過於比較門檻之交易次數，將用於計算勝率
        result['bigger_than_zero'] = result['return'] > 0
        result['bigger_than_benchmark'] = result['return'] > result.benchmark_return

        # 計算與大盤相比之報酬，須高過於大盤報酬才為真正獲利，將用於計算賺賠比
        result['outperformance'] = result['return'] - result.benchmark_return

        # 將各散開的交易表依據交易期間整理在一起，收斂至各交易期間之資訊
        group_result = result.groupby('measure_date')['bigger_than_zero'].agg(
            win_trade_num=('sum'),
            total_trade_num=('count')
        ).reset_index()

        group_result.reset_index(inplace=True)

        # 以下針對以大盤報酬為門檻的方式在執行一次上面動作，並將表合在一起
        group_result = group_result.merge(
            result.groupby('measure_date')['bigger_than_benchmark'].agg(
                win_trade_num_benchmark=('sum')
            ).reset_index(),
            on='measure_date'
        )

        # 計算勝率
        group_result['winning_rate'] = (
            group_result['win_trade_num'] / group_result['total_trade_num']
        )
        group_result['winning_rate_benchmark'] = (
            group_result['win_trade_num_benchmark'] / group_result['total_trade_num']
        )

        # 先前是和各期間各股票報酬的表串，現在則把怕盤報酬串回依交易期間整理後的表
        group_result = group_result.merge(
            date_entry_of_benchmark[['measure_date', 'benchmark_return']],
            on='measure_date'
        )

        # 上述計算完勝率後接著處理賺賠比
        # 將各交易期間之賺賠狀況依據交易期間整理成一張表
        def calculate_average_profit_loss(x):
            return pd.DataFrame([[
                x[x.bigger_than_zero]['return'].mean(),
                x[x.bigger_than_zero != True]['return'].mean()
            ]])

        average_win_loss = result.groupby('measure_date').apply(
            lambda x: calculate_average_profit_loss(x)
        )

        average_win_loss.rename(
            columns={0: 'average_profit', 1: 'average_loss'}, inplace=True
        )
        average_win_loss.reset_index(inplace=True)
        average_win_loss.drop(columns='level_1', inplace=True)

        # 把上述賺賠比結果合併回整理後表格
        final_result = group_result.merge(average_win_loss, on='measure_date')
        final_result.drop(columns=['win_trade_num'], inplace=True)

        # 把賺的狀況為na(當期勝率0，因此沒有賺的交易)的期數填入0，如此一來當期賺賠比會是0
        final_result.average_profit = final_result.average_profit.fillna(0)

        # 計算賺賠比
        final_result['win_loss_ratio'] = (
            final_result.average_profit / abs(final_result.average_loss)
        )

        # 接著計算以大盤報酬為門檻的賺賠比，以下步驟與上述相同
        def calculate_average_profit_loss_benchmark(x):
            return pd.DataFrame([[
                x[x.bigger_than_benchmark]['outperformance'].mean(),
                x[x.bigger_than_benchmark != True]['outperformance'].mean()
            ]])

        average_win_loss_benchmark = result.groupby('measure_date').apply(
            lambda x: calculate_average_profit_loss_benchmark(x)
        )

        average_win_loss_benchmark.rename(
            columns={0: 'average_profit_benchmark', 1: 'average_loss_benchmark'}, inplace=True
        )
        average_win_loss_benchmark.reset_index(inplace=True)
        average_win_loss_benchmark.drop(columns='level_1', inplace=True)

        # 把上述賺賠比結果合併回整理後表格
        final_result = final_result.merge(average_win_loss_benchmark, on='measure_date')
        final_result.drop(columns=['win_trade_num_benchmark'], inplace=True)

        final_result.average_profit_benchmark = final_result.average_profit_benchmark.fillna(0)

        final_result['win_loss_ratio_benchmark'] = (
            final_result.average_profit_benchmark / abs(final_result.average_loss_benchmark)
        )

        # 若勝率為0時，平均賺會是0
        # 若勝率為1時，平均陪便會沒有值(na)，若直接填入0，則會使賺賠比無限大，因此需要剔除此類極端交易
        # 另一種會使賺賠比無限大的狀況是，即使勝率不為1，但剛好平均陪為0者，因此也須將此類交易剃除
        # 其中兩種比較門檻之方式將需要個別衡量
        # 首先比較門檻為0的狀況
        # 清資料
        final_result_clean = final_result[final_result.win_loss_ratio.isna() != True]
        final_result_clean = final_result_clean[final_result_clean.win_loss_ratio != np.inf]

        # 計算數值
        win_loss_ratio_type1 = final_result_clean.win_loss_ratio.mean()
        winning_rate_type1 = final_result_clean.winning_rate.mean()
        expected_return_type1 = (
            winning_rate_type1 * win_loss_ratio_type1 -
            (1 - winning_rate_type1) * 1
        )
        winning_rate_type1 = transform_to_percentage(winning_rate_type1)
        win_loss_ratio_type1 = round(win_loss_ratio_type1, 2)
        expected_return_type1 = round(expected_return_type1, 2)

        # 比較門檻為大盤報酬的狀況
        # 清資料
        final_result_benchmark_clean = final_result[
            final_result.win_loss_ratio_benchmark.isna() != True
        ]
        final_result_benchmark_clean = final_result_benchmark_clean[
            final_result_benchmark_clean.win_loss_ratio_benchmark != np.inf
        ]

        # 計算數值
        win_loss_ratio_benchmark_type1 = final_result_benchmark_clean.win_loss_ratio_benchmark.mean()
        winning_rate_benchmark_type1 = final_result_benchmark_clean.winning_rate_benchmark.mean()
        expected_return_benchmark_type1 = (
            winning_rate_benchmark_type1 * win_loss_ratio_benchmark_type1 -
            (1 - winning_rate_benchmark_type1) * 1
        )
        winning_rate_benchmark_type1 = transform_to_percentage(winning_rate_benchmark_type1)
        win_loss_ratio_benchmark_type1 = round(win_loss_ratio_benchmark_type1, 2)
        expected_return_benchmark_type1 = round(expected_return_benchmark_type1, 2)

        # 以下計算第二種算法
        # 首先計算門檻為0的狀況
        winning_rate_type2 = sum(result.bigger_than_zero) / len(result)

        win_loss_ratio_type2 = (
            result[result.bigger_than_zero]['return'].mean() /
            abs(result[result.bigger_than_zero != True]['return'].mean())
        )

        expected_return_type2 = (
            winning_rate_type2 * win_loss_ratio_type2 -
            (1 - winning_rate_type2) * 1
        )

        winning_rate_type2 = transform_to_percentage(winning_rate_type2)
        win_loss_ratio_type2 = round(win_loss_ratio_type2, 2)
        expected_return_type2 = round(expected_return_type2, 2)

        # 接著計算門檻為benchmark的狀況
        winning_rate_benchmark_type2 = sum(result.bigger_than_benchmark) / len(result)

        win_loss_ratio_benchmark_type2 = (
            result[result.bigger_than_benchmark]['outperformance'].mean() /
            abs(result[result.bigger_than_benchmark != True]['outperformance'].mean())
        )

        expected_return_benchmark_type2 = (
            winning_rate_benchmark_type2 * win_loss_ratio_benchmark_type2 -
            (1 - winning_rate_benchmark_type2) * 1
        )

        winning_rate_benchmark_type2 = transform_to_percentage(winning_rate_benchmark_type2)
        win_loss_ratio_benchmark_type2 = round(win_loss_ratio_benchmark_type2 , 2)
        expected_return_benchmark_type2 = round(expected_return_benchmark_type2, 2)

        # 年化標準差(日年化至年乘以250)
        standard_deviation_benchmark = (
            self.benchmark_data['return'].std() * np.sqrt(annualize_factor)
        )
        standard_deviation_benchmark = transform_to_percentage(standard_deviation_benchmark)
        
        standard_deviation_naive_weight = (
            self.naive_weight_return.net_naive_weight_return.std() * np.sqrt(annualize_factor)
        )
        standard_deviation_naive_weight = transform_to_percentage(standard_deviation_naive_weight)
        
        standard_deviation_cap_weight = (
            self.cap_weight_return.net_cap_weight_return.std() * np.sqrt(annualize_factor)
        )
        standard_deviation_cap_weight = transform_to_percentage(standard_deviation_cap_weight)
        
        # sharpe ratio
        sharp_ratio_benchmark = (
            (self.benchmark_data['return'].mean() * annualize_factor)/
            (self.benchmark_data['return'].std() * np.sqrt(annualize_factor))
        )
        sharp_ratio_benchmark = transform_to_percentage(sharp_ratio_benchmark)
        
        sharp_ratio_naive_weight = (
            (self.naive_weight_return.net_naive_weight_return.mean() * annualize_factor)/
            (self.naive_weight_return.net_naive_weight_return.std() * np.sqrt(annualize_factor))
        )
        sharp_ratio_naive_weight = transform_to_percentage(sharp_ratio_naive_weight)
        
        sharp_ratio_cap_weight = (
            (self.cap_weight_return.net_cap_weight_return.mean() * annualize_factor)/
            (self.cap_weight_return.net_cap_weight_return.std() * np.sqrt(annualize_factor))
        )
        sharp_ratio_cap_weight = transform_to_percentage(sharp_ratio_cap_weight)
        
        # information ratio
        tracking_error_std_naive_weight = np.std(
            naive_weight_and_benchmark_pair.net_naive_weight_return.values -
            naive_weight_and_benchmark_pair['return'].values
        ) * np.sqrt(annualize_factor)
        
        information_ratio_naive_weight = (
            (
                (
                    naive_weight_and_benchmark_pair.net_naive_weight_return.mean() -
                    naive_weight_and_benchmark_pair['return'].mean()
                ) * annualize_factor
            ) / (
                tracking_error_std_naive_weight
            )
        )
        information_ratio_naive_weight = transform_to_percentage(
            information_ratio_naive_weight
        )
        
        tracking_error_std_cap_weight = np.std(
            cap_weight_and_benchmark_pair.net_cap_weight_return.values -
            cap_weight_and_benchmark_pair['return'].values
        ) * np.sqrt(annualize_factor)
        
        information_ratio_cap_weight = (
            (
                (
                    cap_weight_and_benchmark_pair.net_cap_weight_return.mean() -
                    cap_weight_and_benchmark_pair['return'].mean()
                ) * annualize_factor
            ) / (
                tracking_error_std_cap_weight
            )
        )
        information_ratio_cap_weight = transform_to_percentage(
            information_ratio_cap_weight
        )
        
        # 計算alpha, beta之回歸式function
        def run_regression(x, y):
            X = add_constant(x)
            Y = y
            model = OLS(Y, X, missing="drop")
            result = model.fit()
            params = pd.DataFrame(result.params).T
            params.columns = ["Alpha", "Beta"]
            p_value = pd.DataFrame(result.pvalues).T
            p_value.columns = ["Alpha_p_value", "Beta_p_value"]
            
            return params.join(p_value)
        
        regression_result_naive_weight = run_regression(
            naive_weight_and_benchmark_pair['return'].values,
            naive_weight_and_benchmark_pair['net_naive_weight_return'].values
        )
        
        alpha_naive_weight = transform_to_percentage(
            regression_result_naive_weight.Alpha[0]
        )
        alpha_p_value_naive_weight = transform_to_percentage(
            regression_result_naive_weight.Alpha_p_value[0]
        )
        
        beta_naive_weight = transform_to_percentage(
            regression_result_naive_weight.Beta[0]
        )
        beta_p_value_naive_weight = transform_to_percentage(
            regression_result_naive_weight.Beta_p_value[0]
        )
        
        regression_result_cap_weight = run_regression(
            cap_weight_and_benchmark_pair['return'].values,
            cap_weight_and_benchmark_pair['net_cap_weight_return'].values
        )
        
        alpha_cap_weight = transform_to_percentage(
            regression_result_cap_weight.Alpha[0]
        )
        alpha_p_value_cap_weight = transform_to_percentage(
            regression_result_cap_weight.Alpha_p_value[0]
        )
        
        beta_cap_weight = transform_to_percentage(
            regression_result_cap_weight.Beta[0]
        )
        beta_p_value_cap_weight = transform_to_percentage(
            regression_result_cap_weight.Beta_p_value[0]
        )
        
        # 總周轉率
        total_turnover_time_naive_weight = round(
            self.naive_weight_return.turnover_rate.sum(), 2
        )
        total_turnover_time_cap_weight = round(
            self.cap_weight_return.turnover_rate.sum(), 2
        )
        
        # 總交易成本
        total_transaction_cost_naive_weight = transform_to_percentage(
            self.naive_weight_return.transaction_cost_rate.sum()
        )
        total_transaction_cost_cap_weight = transform_to_percentage(
            self.cap_weight_return.transaction_cost_rate.sum()
        )

        table_header = ['Backtesting result', 'Benchmark', 'Naive weight', 'Cap weight']

        # 平常不會用這樣的indentation，但為了方便看所以使用以下方式
        table_data = [
            ('Start date',
            start_date, start_date, start_date),
            ('End date',
            end_date, end_date, end_date),
            ('Duration (years)',
            trading_years, trading_years, trading_years),
            ('Total trade',
            'NA', total_trade_num, total_trade_num),
            ('Total entry times',
            'NA', total_trade_period, total_trade_period),
            ('Average trade number per entry',
            'NA', trade_num_per_period, trade_num_per_period),
            ('Average holding days',
            'NA', holding_days_per_period, holding_days_per_period),
            ('Cumulative return',
            cumulative_return_benchmark, cumulative_return_naive_weight, cumulative_return_cap_weight),
            ('Annual return',
            annual_return_benchmark, annual_return_naive_weight, annual_return_cap_weight),
            ('CAGR',
            CAGR_benchmark, CAGR_naive_weight, CAGR_cap_weight),
            ('Historical high of equity',
            max_cumulative_return_benchmark, max_cumulative_return_naive_weight, max_cumulative_return_cap_weight),
            ('Historical low of equity',
            min_cumulative_return_benchmark, min_cumulative_return_naive_weight, min_cumulative_return_cap_weight),
            ('Maximum drawdown',
            max_drawdown_benchmark, max_drawdown_naive_weight, max_drawdown_cap_weight),
            ('Total return over max drawdown',
            total_return_over_max_drawdown_benchmark, total_return_over_max_drawdown_naive_weight, total_return_over_max_drawdown_cap_weight),
            ('Maximum outperformance',
            'NA', max_outperformance_naive_weight, max_outperformance_cap_weight),
            ('Minimum underperformance',
            'NA', min_outperformance_naive_weight, min_outperformance_cap_weight),
            ('Maximum under water days',
            max_under_water_days_benchmark, max_under_water_days_naive_weight, max_under_water_days_cap_weight),
            ('Winning rate (portfolio)',
            winning_rate_benchmark, winning_rate_naive_weight, winning_rate_cap_weight),
            
            # 以下為新增部分，主要計算勝率、賺賠比、期望值
            # 由於有兩種比較門檻(0, benchmark)與兩種衡量方法(依據各交易期間、所有一起衡量)
            # 同時每種方法又有3個數值需要報告，因此總共會呈現4 x 3 =12個數值
            # 特別空出來，若不需要可以縮減
            ('Measured by each period-individual stock',
             '', '', ''),
            ('Winning rate (individual stock)',
            'NA', winning_rate_type1, winning_rate_type1),
            ('Win loss ratio',
            'NA', win_loss_ratio_type1, win_loss_ratio_type1),
            ('Expected return',
            'NA', expected_return_type1, expected_return_type1),
            
            ('Winning rate (compare to benchmark)',
            'NA', winning_rate_benchmark_type1, winning_rate_benchmark_type1),
            ('Win loss ratio (compare to benchmark)',
            'NA', win_loss_ratio_benchmark_type1, win_loss_ratio_benchmark_type1),
            ('Expected return (compare to benchmark)',
            'NA', expected_return_benchmark_type1, expected_return_benchmark_type1),

            ('Measured by whole period-individual stock',
             '', '', ''),
            ('Winning rate (individual stock)',
            'NA', winning_rate_type2, winning_rate_type2),
            ('Win loss ratio',
            'NA', win_loss_ratio_type2, win_loss_ratio_type2),
            ('Expected return',
            'NA', expected_return_type2, expected_return_type2),

            ('Winning rate (compare to benchmark)',
            'NA', winning_rate_benchmark_type2, winning_rate_benchmark_type2),
            ('Win loss ratio (compare to benchmark)',
            'NA', win_loss_ratio_benchmark_type2, win_loss_ratio_benchmark_type2),
            ('Expected return (compare to benchmark)',
            'NA', expected_return_benchmark_type2, expected_return_benchmark_type2),
            
            ('Standard deviation (annualized)',
            standard_deviation_benchmark, standard_deviation_naive_weight, standard_deviation_cap_weight),
            ('Sharpe ratio (annualized)',
            sharp_ratio_benchmark, sharp_ratio_naive_weight, sharp_ratio_cap_weight),
            ('Information ratio (annualized)',
            'NA', information_ratio_naive_weight, information_ratio_cap_weight),
            ('Alpha',
            'NA', alpha_naive_weight, alpha_cap_weight),
            ('Alpha\'s p value', 
            'NA', alpha_p_value_naive_weight, alpha_p_value_cap_weight),
            ('Beta',
            'NA', beta_naive_weight, beta_cap_weight),
            ('Beta\'s p value',
            'NA', beta_p_value_naive_weight, beta_p_value_cap_weight),
            ('Total turnover times',
            'NA', total_turnover_time_naive_weight, total_turnover_time_cap_weight),
            ('Total transaction cost',
            'NA', total_transaction_cost_naive_weight, total_transaction_cost_cap_weight)
        ]
        print(tabulate(table_data, headers=table_header, tablefmt='rst'))

        table_data = pd.DataFrame(table_data)
        dataframe_index = table_data.iloc[:, 0]
        table_data.drop(columns=0, inplace=True)
        table_data.index = dataframe_index
        table_data.index.name = 'Backtesting result'
        table_data.columns = ['Benchmark', 'Naive weight', 'Cap weight']
        self.table_data = table_data