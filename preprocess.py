import pandas as pd
import numpy as np
from datetime import datetime
import operator
from itertools import compress
import math


class Data:
    def __init__(
        self,
        data_name,
        data_frequency,
        CSR=True,
        market_size_threshold=None,
        market_size_select_rank=None,
        market_size_select_number=None,
        market_size_select_ratio=None
    ):
        '''
        讀入資料，每個資料本身都是一個Data物件，此物件帶有許多屬性與功能，主要用於撰寫選股策略。

        Parameters
        ----------
        data_name:
            資料名稱
        data_frequency:
            指定資料頻率
            - year: 年
            - season: 季
            - month: 月
            - day: 天
        market_size_threshold:
            選擇大過於給定市值門檻之股票，單位為億(int, float)，
            若給定數值，則將忽略後續透過絕對家數或比例之條件，
            屬於市值篩選條件中最優先考量者
        market_size_select_rank:
            需搭配market_size_select_number, market_size_select_ratio使用，
            - top: 市值由大至小排序
            - bottom: 市值由小至大排序
            接著透過market_size_select_number, market_size_select_ratio選擇
        market_size_select_number:
            依據market_size_select_rank之排序選擇固定之股票家數(int)
        market_size_select_ratio:
            依據market_size_select_rank之排序選擇固定比例之家數(int)

        Description
        -----------                                    
        市值門檻之篩選優先於所有篩選條件(add_filter, select)
        當執行add_filter或是select時，皆會將市值條件納入考量，說明：
        1. 若執行add_filter則會找出給定filter條件及市值條件之交集
        2. 若只執行select則會先通過市值門檻才進行排序篩選
        3. 若先執行add_filter接著執行select，則會依據通過add_filter篩選條件且通過市值門檻者
           才進行select的排序篩選，因此每期之樣本群在此情況下會有所差異
        
        Example
        -------
        選擇市值前200者，接者選出給定財務資訊top 30%者，則篩選結果會是每期60支個股，
        但若在中間加入另一個篩選條件如：財務資訊數值需大於0，則進到select前之樣本便不會是每期200支，
        因此每期select執行後，選出來的個股數便不會是每期60支。
        '''
        self.data_name = data_name
        self.market_size_threshold = market_size_threshold
        self.market_size_select_rank = market_size_select_rank
        self.market_size_select_number = market_size_select_number
        self.market_size_select_ratio = market_size_select_ratio
        try:
            
            self.data = pd.read_csv('data/{}.csv'.format(self.data_name), index_col=0, encoding='cp950')
            
            if CSR and self.data_name != 'benchmark':
                # CSR pool select, if CSR pool select we need to read the excel file
                # Then filter out the stock we need.
                stock_list = pd.read_excel('112年8月勞退CSR可投資標的.xlsx')
                stock_list = stock_list['Code']
                stock_list = stock_list.tolist()
                checkdata = self.data[self.data.index.isin(stock_list)]
                self.data = checkdata
                
                
        except Exception as e:
            print('「{}」 不存在!'.format(self.data_name))
            print("請輸入正確資料名稱，例如：稅後純益成長率(%)_已格式化")
            print('Error: {}'.format(e))
        self.data_frequency = data_frequency
        # 如果資料週期是週的話要例外處理

        # 找尋相對應的mapping表
        if self.data_frequency == 'season':
            self.reference_timeframe = season_timeframe
        if self.data_frequency == 'month':
            self.reference_timeframe = month_timeframe
        # 建立將data_frequency轉為數字排序，回測產生訊號時會使用到
        data_frequency_list = {'year': 1, 'season': 2, 'month':3,'week':4, 'day':5}
        self.__data_frequency_num = data_frequency_list[self.data_frequency]
        
        # 因為 FactSET的資料有些會有Cmoney沒有的所以要利用Cmoney的資料將FactSet多的資料刪掉
        if self.data_frequency == 'week':
            week_actual_date = pd.read_csv('week_date.csv',header=None)
            
            week_actual_date = list(week_actual_date.iloc[:,0])
            #
            week_actual_date = list(map(str, week_actual_date))
            check_data = self.data.loc[:,self.data.columns.isin(week_actual_date)]
            self.data = check_data
        
        # 進行mapping，找尋離公布日最近之股票
        # 會是同一天或是公布日的上一個交易日(若公布當日沒有交易日)
        # 這是當資料的頻率為月以上的時候才會去考慮的(而且只有去服務季報跟月營收的資料)
        if self.data_frequency != 'day' and self.data_frequency != 'week':
            self.data.columns = list(map(
                lambda x: datetime.strptime(self.reference_timeframe[x], '%Y%m%d'),
                self.data.columns
            ))
            possible_trade_date = []
            for date in self.data.columns:
                temp_index = actual_trade_time >= date
                temp_target = actual_trade_time[temp_index]
                # 以下try和except捕捉以下可能狀況，例如：
                # 6月之營收最後公布日為7/10，但資料庫更新時，有些個股可能會提前公布
                # (因此6月之營收資料已存在，即使7/10還沒到)
                # 此時若股價資料只更新到7/5，就會出現error
                # 因為即使提前公布，程式仍會假定此資料之公布日為7/10
                # 如此一來上述的temp_index則會為None，temp_target也同為None
                # 進到try中的min時，則會出現error，因為None無法取min
                # 下面將會把columns為na的欄位drop掉
                # 這樣也符合邏輯，畢竟一定要等到10號規定之日期後，資料到齊才能進行全部的比較
                try:
                    possible_trade_date.append(min(temp_target))
                except:
                    possible_trade_date.append(None)
            self.data.columns = possible_trade_date
        else:
            self.data.columns = list(map(
                lambda x: datetime.strptime(x, '%Y%m%d'),
                self.data.columns
            ))

        
        # 把columns為na的欄位drop掉
        self.data = self.data[
            self.data.columns[pd.notna(self.data.columns)]
        ]
        
        # 儲存公布日期
        self.announce_date = self.data.columns

        # 建立使用select的計數器
        # 更改add_filter中的篩選條件(自定義function)或select中的不同排名選取條件時
        # 都必須重新執行add_filter才能確保Data class的訊號正確
        # 因為內部signal皆採取直接迭代方式，若沒有重新執行add_filter
        # 則內部訊號將會為上次篩選條件後的殘餘結果，如此一來便會有錯
        # 因此透過建立__select_function_usage_counter來判斷是否有上述會產生錯誤的情形出現
        self.__select_function_usage_counter = 0

        # 以下會透過self.is_piped來判斷資料是否有被pipe過，詳細內容可參考下面的pipe function
        self.is_piped = False

    def mutate_data(
        self,
        target_function,
        rebalance_frequency='week',
        date_order_ascending=False,
        measure_on_nth_date = 1
    ):
        '''
        對於給定資料進行數值計算(加工)，同時可以決定再平衡的衡量時點。

        Parameters
        ----------
        target_function:
            欲對各股票計算數值之function，將會對一開始輸入之資料進行此動作，
            因此所有初始資料將會被用於計算。也可以填入None，則代表不對資料做計算，
            但仍然可以對出場日做指定。
            此target_function在底層主要透過np.apply_along_axis()執行，
            因此會依據每一個row去apply輸入的function，在設計function時只要掌握此邏輯，
            便可以知道該如何設計欲apply的function。
        rebalance_frequency:
            欲進行投資組合再平衡的頻率，可選擇頻率如下：
            - year: 年
            - quarter : 季
            - month: 月
            - week: 週
        date_order_ascending:
            將上述再平衡頻率週期的日期小至大排序(True)，或大致小排序(False)。
        measure_on_nth_date:
            依據給定rebalance_frequency及date_order_ascending之排序，
            選取定義好之時間窗格下的第n筆交易日進行再平衡判斷(所以會在n+1出場)，
            若給定數值大過於當個資料窗格下之總交易日數，則會選擇給定排序後的資料窗格下以
            最後一個交易日作為出場日。

        Description
        -----------
        執行後將會依據給定之rebalance_frequency進行資料縮減，因為回測之再平衡日皆是由資料
        帶有之日期進行驅動，因此只會保留實際上用於再平衡衡量時點的資料。

        Example
        -------
        給定rebalance_frequency='week', date_order_ascending='False', 
        measure_on_nth_date='1'，假設target_function為計算近20筆資料之平均，
        而輸入之資料為收盤價，此時target_function執行後將算出每天的20日移動平均，
        並迭代回原先資料，接著依據此筆資料既有之日期，找出欲判斷出場的日期，
        在上述給定參數下，找尋出場日期便會將日期資料依據週整理好(groupby)，
        接著依據大致小排序各週內之日期，並選擇最上方之第1個日期作為衡量出場日的時點，
        如此一來，便能以當週最後一筆交易日作為再平衡時點判斷日，接著以t+1日(下週的第一天)建立部位。
        '''
        self.rebalance_frequency = rebalance_frequency
        self.date_order_ascending = date_order_ascending
        self.measure_on_nth_date = measure_on_nth_date
        # 透過np的apply速度較快，但columns跟index都會不見，所下方重新指定回去
        if target_function != None:
            mutated_data = pd.DataFrame(
                np.apply_along_axis(target_function, 1, self.data.values)
            )
            mutated_data.columns = self.data.columns
            mutated_data.index = self.data.index
            self.data = mutated_data
            print('資料修改 - \'{}\' 已完成!'.format(self.data_name))

        # 接著創立出場日期並選取相對應資料
        # 依據日期建立其詳細資訊將用於決定出場時點(年、季、月、週、星期幾)
        date_information = np.array(list(map(
            lambda x: [x.year, x.quarter, x.month, x.week, x.dayofweek+1],
            self.data.columns
        )))

        date_information = pd.DataFrame(
            date_information,
            columns=['year', 'quarter', 'month', 'week', 'dayofweek']
        )

        date_information['date'] = self.data.columns

        # 建立function選取出場日期，將應用在對date_information做groupby.apply()中
        def select_exit_date(
            sliced_data, ascending, measure_on_nth_date
        ):
            # 為了讓使用上更直覺，所以使用者輸入之邏輯為1開始，但轉入python則須-1，才能轉換成由0開始
            measure_on_nth_date = measure_on_nth_date - 1
            if len(sliced_data) <= measure_on_nth_date:
                measure_on_nth_date = len(sliced_data)-1
            elif measure_on_nth_date < 0:
                raise Exception('measure_on_nth_date之輸入不得小於1')
            
            return sliced_data.sort_values(
                'date', ascending=ascending
            ).iloc[measure_on_nth_date, :]

        # 以下便可以彈性選擇最終欲出場之日期
        target_exit_date = date_information.groupby(
            ['year', self.rebalance_frequency]
        ).apply(
            lambda x: select_exit_date(
                sliced_data=x,
                ascending=self.date_order_ascending,
                measure_on_nth_date=self.measure_on_nth_date
            )
        )
        # 前面加雙底線主要用於內部使用，外部無法直接呼叫
        self.__target_exit_date_index = list(map(
            lambda x: x in target_exit_date.date.values,
            self.data.columns.values
        ))
        # 儲存所有後續可能需要用到的資訊
        # 選擇依據出場日期指定的data
        self.data = self.data.iloc[:, self.__target_exit_date_index]
        # 儲存出場日期
        self.target_exit_date = target_exit_date
        # 將announce_date縮減至僅出場日期，否則後續會有index out of range的問題
        self.announce_date = list(compress(
            self.announce_date,
            self.__target_exit_date_index
        ))
        # 建立dict儲存相對應的轉換中文，為了讓print出來時是以中文表達
        rebalance_frequency_dict = dict({
            'year': '年', 'quarter': '季', 'month': '月', 'week': '週'
        })
        date_order_ascending_dict = dict({True: '第', False: '最後'})
        print(
            '出場衡量 - \'每{}{}{}個交易日進行再平衡衡量(於下一個交易日出場)\' 已指定!'.format(
                rebalance_frequency_dict[self.rebalance_frequency],
                date_order_ascending_dict[self.date_order_ascending],
                self.measure_on_nth_date
            )
        )
        print(' ')
    
    def pipe_to(
        self,
        pipe_to_data
    ):
        '''
        將自身訊號的條件傳遞給下一個data中。

        Parameters
        ----------
        pipe_to_data:
            在自身data處理並產生訊號後，接著欲傳遞至的data對象。

        Description
        -----------
        主要用法是在接續考量兩data間之關係，直接以例子敘述用途：
        假設今天想要在滿足data_1的給定條件後，接著依據data_2的數值進行排序選擇，
        例如：須滿足波動度(月)小於0.3者，接著依據月營收yoy排序找尋最高之30%者，
        程式用法將會是，先將波動度資料(Data物件)透過add_filter產生訊號後，
        透過pipe把訊號送到月營收yoy(Data物件)，此時執行pipe_to後，
        月營收yoy帶有之訊號便會是符合波動度之訊號(交集)，
        因此，月營收yoy帶有的訊號，便會是符合波動度小於0.3者之月營收yoy財務數值，
        執行程式後會回傳月營收yoy的Data物件，
        接著便可以繼續執行select功能選擇月營收yoy最高之30%者，
        如此一來，最終結果便會是符合波動度小於0.3者後，接著依據月營收yoy排序選擇最高之30%者。

        日期處理方面也可以由不同資料頻率搭配，例如：
        季頻率資訊可能搭配月頻率資訊，因此兩季之間可能含有月資料，
        所以要找出落於date_entry, date_exit之間的可交易日與符合篩選條件之個股，
        同時使用季與月資料時，如5/15~8/14之間含有6/10, 7/10, 8/10之月資訊公告，
        此時main_tradable_set則為5/15當時，符合季篩選條件之個股清單，
        而sub_tradable_set則為各月營收公告時，符合月篩選條件之個股清單(以上述為例則有3份清單)，
        接者將main_tradable_set與sub_tradable_set取交集(結果會有三分個股清單)，
        若是處理同頻率之資料，情況則沒有上述那麼複雜，純粹取同頻率之交集。

        Notes
        -----
        1. 自身(self)data之頻率要低過於pipe_to_data之頻率，
           由於最終會以pipe_to_data之資料頻率為主，因此若頻率顛倒，
           在邏輯上會很奇怪(反而由高頻資料去交易低頻)，
           此處處理的邏輯類似Backtest class裡處理訊號的方式
        2. 執行pipe_to後回傳的會是Data物件，因此可以繼續使用select, add_filter等功能
        3. 若self.data的市值限制與pipe_to_data的市值限制有差異，其實也是可行的，
           但須確保self.data之市值限制鬆過於pipe_to_data之市值限制，
           因為當self.data的訊號傳遞給pipe_to_data時，代表已通過self.data的市值門檻，
           此時若pipe_to_data的市值門檻鬆過於self.data則不會起任何作用，
           因為還是會以較嚴格之self.data的市值門檻為主
        '''
        # 首先確認pipe_to_data是否帶有signal，以及是否執行過市值篩選
        # 因為下面會依據signal進行處理，所以必須先帶有signal
        # 若未曾執行過add_filter或是select則將所有data寫入signal
        if hasattr(pipe_to_data, 'signal') != True:
            pipe_to_data_signal = {}
            for i in range(len(pipe_to_data.data.columns)):
                date = pipe_to_data.data.columns[i]
                pipe_to_data_signal[date] = dict(zip(
                    pipe_to_data.data.index, pipe_to_data.data.iloc[:, i]
                ))
            pipe_to_data.signal = pipe_to_data_signal
            # 接著執行市值篩選條件
            # 在此情況下可確保pipe_to_data已帶有signal因此可以執行__market_size_filter()
            pipe_to_data.__market_size_filter()



        # 建立空的signal來裝整理後的資訊結果
        piped_signal = {}
        for i in range(len(self.signal.keys())):
            # 避免當i為最後一個值時會出現'index out of range'的問題
            if i != len(self.signal.keys()) - 1:
                # 依據本身訊號找出進出場區間
                date_entry = list(self.signal.keys())[i]
                date_exit = list(self.signal.keys())[i+1]
            main_tradable_set = set(self.signal[date_entry].keys())

            # 透過index找出符合訊號區間之日期
            index = list(map(
                lambda x: (x >= date_entry) & (x < date_exit),
                list(pipe_to_data.signal.keys())
            ))
            trade_date = list(compress(pipe_to_data.signal.keys(), index))

            # 在給定日期下，接著找出pipe_to_data中符合self.signal訊號者
            for date in trade_date:
                sub_tradable_set = set(pipe_to_data.signal[date].keys())
                qualified_tradable_set = main_tradable_set.intersection(
                    sub_tradable_set
                )

                # 暫時裝結果，依據資料透過迴圈寫入dict中
                temp_piped_signal = {}
                for key in qualified_tradable_set:
                    temp_piped_signal[key] = pipe_to_data.data[date][key]

                # 迭代回最終結果，此時pipe_to_data的訊號已和self.data取過交集
                piped_signal[date] = temp_piped_signal
        pipe_to_data.signal = piped_signal
        print(
            '\'{}\' 之訊號已成功傳遞至 \'{}\''.format(
                self.data_name, pipe_to_data.data_name
            )
        )
        # 執行pipe後將被pipe的資料之is_piped 設為True作為判斷
        # 若之後需要執行add_filter則會用到
        # 若曾被pipe過則需要使用self.signal作為filter的判斷，因為要保留pipe後的結果
        # 若未曾被pipe則直接使用原先的data資料
        pipe_to_data.is_piped = True

        # 直接回傳pipe_to_data之目的在於可以在程式後面直接select或add_filter等功能
        # 在程式表達上會更為直覺，當然也可以拆成兩行去寫，但仍會回傳pipe_to_data物件
        return pipe_to_data

    # function前面加雙底線為內部使用function，物件建立後由外部無法直接呼叫
    # (仍有辦法由外部呼叫，但呼叫方式與一般使用方法不同)
    def __market_size_filter(self):
        # 依據給定訊號去檢查該訊號對應之日期是否有通過市值門檻，因為後續若使用到add_filter, select功能時
        # 必須要先通過市值條件(最上層條件)後才能進行篩選(select功能)
        # 建立空的dict名為qualified_market_size_signal儲存依據給定signal下
        # 對應signal日期，有通過市值篩選之股票代碼與市值資料
        # 除了代碼外仍保存市值資料原因是分析報酬時使用cap_weight時會用到
        # qualified_market_size_signal: {'siganl_time': {company_codes: 該股當期市值}, ...}

        # 這個inner function會在add_filter與select中被呼叫，且只會被呼叫一次
        # 因為signal只會在add_filter或select被執行後才會產生，其中select在一開始執行前會有檢查機制
        # 會先檢查signal屬性是否存在，若已存在則不再執行本inner function，若不存在則先執行，接著才進行排序選擇

        # 主要應用邏輯是取出符合市值篩選條件與訊號本身之交集
        # 若曾執行add_filter則代表需要同時符合filter條件與市值篩選條件
        # 若未曾執行add_filter，則代表在既有資料下單純找出符合市值篩選條件者(接著進行select排序選擇)
        # 若執行add_filter後接著執行select，此時的樣本則已縮減至同時通過市值與add_filter此兩層條件後才接續進行排序篩選
        qualified_market_size_signal = {}
        for date_key in self.data.columns:
            # 確保給定signal欲判斷的日期需要涵蓋在市值既有之日期中
            if date_key >= market_size_inner_usage.data.columns[0]:
                # 依據日期抓出欲檢查的市值資料，如此一來只要比較特定日期即可

                try:
                    target_market_size_data = market_size_inner_usage.data[date_key]
                    
                # 基本上由於市值資料涵蓋所有交易日期，因此signal帶有的日期應該接包含於其中
                # 但若出現market_size的尾段日期少過於signal欲比較日期時，便會出現error
                # 此種狀況為市值資料更新不足，亦即欲比較signal之資料日期多過於市值
                except Exception as e:
                    print('市值資料可能需要更新！')
                    print('Error: {}'.format(e))
                    print(' ')
                    break

                # 篩選符合市值條件者，門檻值會優先被考量，因此只要market_size_threshold有填入值
                # 其他市值篩選條件設定將被忽略
                if self.market_size_threshold != None:
                    qualified_market_size_signal[date_key] = dict(zip(
                        target_market_size_data[
                            target_market_size_data > self.market_size_threshold
                        ].index,
                        target_market_size_data[
                            target_market_size_data > self.market_size_threshold
                        ]
                    ))
                else:
                    # 確保市值篩選條件正確
                    assert (self.market_size_select_number==None)|(self.market_size_select_ratio==None), 'number與ratio無法同時選擇'
                    assert (self.market_size_select_number!=None)|(self.market_size_select_ratio!=None), '請擇一輸入number與ratio之條件'
                    
                    if self.market_size_select_rank == 'top':
                        ascending=False
                    elif self.market_size_select_rank == 'bottom':
                        ascending=True
                    else:
                        print('market_size_select_rank請輸入top或是bottom')
                        break
                    
                    if self.market_size_select_number != None:
                        select_number = self.market_size_select_number
                    else:
                        # 依據給定日期，將市值na值去除後才進行比例選擇，因此每期之樣本數會有差異
                        select_number = round(
                            len(market_size_inner_usage.data[date_key].dropna())*self.market_size_select_ratio
                        )
                    # 依據日期寫入通過市值條件之股票代碼
                    filtered_result = target_market_size_data.sort_values(ascending=ascending).iloc[:select_number, ]
                    qualified_market_size_signal[date_key] = filtered_result
        
        if self.market_size_threshold != None:
            print(
                '市值條件 - \'{}億以上\' 門檻已選擇！'.format(self.market_size_threshold)
            )
        elif self.market_size_select_number != None:
            print(
                '市值條件 - \'{} {}\' 名已選擇！'.format(
                    self.market_size_select_rank, self.market_size_select_number
                    )
            )
        elif self.market_size_select_ratio != None:
            print(
                '市值條件 - \'{} {}\' 比例已選擇！'.format(
                    self.market_size_select_rank, self.market_size_select_ratio
                    )
            )
        print(' ')
        # 把通過市值門檻的公司存起來，分析報酬時使用cap_weight時會用到
        self.qualified_market_size_signal = qualified_market_size_signal

        # 建立空的dict來裝通過市值門檻的股票及其當期財務資訊
        filtered_result = {}
        
        for date_key in self.signal.keys():
            # 以同時滿足市值條件之日期為主(因為可能在給定日期下signal有資料而市值沒有)
            if date_key in self.qualified_market_size_signal.keys():
                # 若皆有日期才寫入，也需要先創立空的set否則下方會因為沒有建立好schema而出現error
                filtered_result[date_key] = {}
                for code in self.signal[date_key].keys():
                    # self.qualified_market_size_signal[date_key]為給訂日期下通過市值門檻之股票
                    if code in self.qualified_market_size_signal[date_key].keys():
                        # self.signal[date_key][code]為對應個股之財務數值，已通過篩選條件及市值門檻
                        filtered_result[date_key][code] = self.signal[date_key][code]
        # 將已通過篩選條件與市值門檻之股票迭代回原先之self.signal
        self.signal = filtered_result

    def add_filter(
        self,
        condition_filter
    ):
        '''
        對於既有data執行condition_filter的篩選。

        Parameters
        ---------
        condition_filter:
            欲篩選之條件(function)，此function會對data中所有值做判斷。

        Notes
        -----
        1. add_filter必須在select之前被執行(同一個data下)，
           若先執行select才接著執行add_filter，由於add_filter在沒有用過pipe下，
           只會依據self.data做篩選，因此若先執行select產生signal後，
           此signal結果仍然不會被納入考量，因為在add_filter中會依據self.data去判斷,
           並產生signal蓋掉原先select後產生的signal，因此先執行select則沒有任何意義
        '''
        # self.signal: {'siganl_time': {company_code: 相對應的財務數值,
        #                               company_code: 相對應的財務數值}, ...}

        # 若曾被執行pipe(接收訊號)，則is_piped會被標記為True
        # 此時self本身便會帶有訊號(signal)，處理方式便會不同
        # 若曾被pipe則需要使用signal作為filter的判斷，因為要保留pipe後的結果
        # 若未曾被pipe則直接使用原先的data資料
        if self.is_piped == True:
            try:
                # 產生空的signal裝結果
                signal = {}
                for date in self.signal.keys():
                    # 先產生set裝結果
                    signal[date] = {}
                    for code in self.signal[date].keys():
                        if condition_filter(self.signal[date][code]) == True:
                            signal[date][code] = self.signal[date][code]
                self.signal = signal
            except Exception as e:
                print('使用之篩選條件有問題!')
                print('Error: {}'.format(e))
                print('')
        else:
            try:
                condition_filter_result = list(map(
                    lambda x: condition_filter(x),
                    self.data.values
                ))
                
                condition_filter_result = np.array(condition_filter_result)
                # 處理訊號，依據日期裝入dict，各日期中含有公司代碼與數值的dict
                signal = dict()
                for i in range(len(self.announce_date)):
                    date = self.announce_date[i]
                    temp_signal = condition_filter_result[:, i]
                    temp_data = self.data.values[:, i]
                    code_index = np.where(temp_signal == True)
                    
                    signal[date] = dict(zip(
                        self.data.index[code_index],
                        temp_data[code_index]
                    ))
                self.signal = signal
                print('篩選條件 - \'{}\' 已新增!'.format(condition_filter.__name__))
            except Exception as e:
                print('使用之篩選條件有問題!')
                print('Error: {}'.format(e))
                print('')

        # 執行市值篩選條件
        # self.__select_function_usage_counter會加一的情況只有在select功能中
        # 而add_filter可以將既有之訊號蓋掉，因此有歸零之效果，可確保訊號正確性
        # (但要確保若有經過mutate後的資料是自己想要的，因為mutate後的資料或被縮減)
        # 因此若是策略完全更改，則重新輸入資料較安全
        self.__market_size_filter()
        if self.__select_function_usage_counter == 1:
            self.__select_function_usage_counter -= 1

    def select(
        self,
        rank='top',
        number=None,
        ratio=None
    ):
        '''
        依據給定排序選擇絕對數量或比例。

        Parameters
        ----------
        rank:
            - top：選擇排序較大數值
            - bottom：選擇排序較小的數值
        number:
            選擇家數上限(int)
        ratio：
            選擇家數比例(float)

        Description
        -----------
        所有資料在被選擇前都會通過市值條件之篩選。

        Notes
        ------
        1. number與ratio只能同時存在一個條件
        '''
        # 若沒有執行add_filter則不會產生signal，因此先判斷signal是否已存在
        # 若signal不存在則將既有的值皆加入signal當中
        if hasattr(self, 'signal') != True:
            signal = {}
            for i in range(len(self.data.columns)):
                date = self.data.columns[i]
                signal[date] = dict(zip(
                    self.data.index, self.data.iloc[:, i]
                ))
            self.signal = signal
            # 接著執行市值篩選條件
            self.__market_size_filter()

        # 運行到這可以確保data皆已通過市值篩選
        # 若self曾執行過add_filter，便會帶有訊號同時已在add_filter中執行過__market_size_filter()
        # 以下接著進行篩選
        # 須注意，若曾執行add_filter後接著執行select
        # 此時的樣本會縮減至同時通過市值與add_filter此兩層條件後才接續進行排序篩選
        # 因此每期樣本數會有所差異，例如
        # 選擇市值前200者，接者選出給定財務資訊top 30%者，則篩選結果會是每期60支個股
        # 但若在中間加入另一個篩選條件如：財務資訊數值需大於0，則進到select前之樣本便不會是每期200
        # 因此每期select執行後，選出來的個股數便不會是每期60支

        # 確保篩選條件正確
        assert (number==None)|(ratio==None), 'number與ratio無法同時選擇'
        assert (number!=None)|(ratio!=None), '請擇一輸入number與ratio之條件'
        # 指定後續排序參數
        if rank == 'top':
            reverse = True
        elif rank == 'bottom':
            reverse = False
        else:
            raise Exception('未知的篩選條件，請輸入top或bottom')
        # 處理訊號篩選
        final_filter_result = {}
        for key in self.signal.keys():
            # 需要先把dict中nan的值drop掉(包含key)，才不會有問題
            # 可參考：https://stackoverflow.com/questions/4240050/python-sort-function-breaks-in-the-presence-of-nan
            # 在debug時發現上述的問題，若nan沒去除，將會依據nan為斷點產生序列區間，針對每個區間作排序
            # 這樣最後在取top或tail排序值時就會有問題，因為接續的值沒有被納入整體排序
            # 以下寫法可參考：https://stackoverflow.com/questions/26323080/filtering-null-values-from-keys-of-dictionary-python
            self.signal[key] = dict(
                (dict_key, dict_value) for dict_key, dict_value in self.signal[key].items()
                if not math.isnan(dict_value)
            )

            # 清完na值之後才能知道真正可排序之公司數量，否則會有高估選取數量的問題
            # (因為沒清na前數量一定較多，若依據此選出比例，在清完na後數量較少情況下，則出現高估選取數量的問題)
            if number != None:
                select_number = number
            else:
                select_number = round(len(self.signal[key])*ratio)

            # 清完nan值後才開始排序
            final_filter_result[key] = dict(
                sorted(
                    self.signal[key].items(),
                    key=operator.itemgetter(1),
                    reverse=reverse
                )[: select_number]
            )
        self.signal = final_filter_result
        if number != None:
            print(
                '{} - \'{}{}{}\' 名已選擇!'.format(self.data_name, rank, ' ', number)
            )
        else:
            print(
                '{} - \'{}{}{}\' 比例已選擇!'.format(self.data_name, rank, ' ', ratio)
            )
        print('')
        
        self.__select_function_usage_counter += 1
        if self.__select_function_usage_counter >= 2:
            self.__select_function_usage_counter = 0
            raise Exception('請重新執行add_filter，否則訊號將有錯誤!')


# 以下一些變數只定義於此py檔中，目的在於供主要class使用
# 在另外一個notebook中若只import欲使用之class
# 則以下變數在當個notebook中將不會存在


# 建立需要mapping的表，提供上述Data class使用
# 找尋實際有交易的日期(開市)，依據開盤價之日期作為判斷
# 若欲使用收盤價作為進出場價在此處也不會受影響
# 因為只是做為判斷是否有交易日
open_price = pd.read_csv('data/open_price.csv', index_col=0, encoding='cp950')


    

actual_trade_time = list(map(
    lambda x: datetime.strptime(x, '%Y%m%d'),
    open_price.columns
))
actual_trade_time = pd.DataFrame(actual_trade_time)[0]

# 季報公布日期
# 2013以前，財報公布日期→4/30, 8/31, 10/31, 3/31
fin_stat_mapping_date_old = dict({
    1: '0430',
    2: '0831',
    3: '1031',
    4: '0331'
})

# 2014以後，財報公布日期→5/15, 8/14, 11/14, 3/31
fin_stat_mapping_date_new = dict({
    1: '0515',
    2: '0814',
    3: '1114',
    4: '0331'
})

# mapping季報表對應之日期
season_timeframe = dict()
for year in range(2000, 2030):
    for season in range(1, 5):
        if year <= 2013:
            target_mapping_date = fin_stat_mapping_date_old
        else:
            target_mapping_date = fin_stat_mapping_date_new
        if season == 4:
            mapped_date = target_mapping_date[season]
            mapped_date = str(year+1) + mapped_date
            key = str(year) + str(0) + str(season)
            season_timeframe[key] = mapped_date
        else:
            mapped_date = target_mapping_date[season]
            mapped_date = str(year) + mapped_date
            key = str(year) + str(0) + str(season)
            season_timeframe[key] = mapped_date

# mapping月資訊對應之日期(每月10號公布)
month_timeframe = dict()
for year in range(2000, 2030):
    for month in range(1, 13):
        if month < 10:
            key = str(year) + str(0) + str(month)
        else:
            key = str(year) + str(month)
        if month < 12:
            mapped_date = str(year) + str(month+1) + str(10)
        else:
            mapped_date = str(year+1) + str(1) + str(10)
        month_timeframe[key] = mapped_date


# market_size_inner_usage為上述Data內部使用之變數，也是Data class之instance
# 目的是要能夠提供訊號產生過程時進行市值條件比對
# 因為在Data class中會使用到market_size_inner_usage(Data class之instance)本身
# 而Data class與market_size_inner_usage在此py檔中類似同時建立的關係(但仍有前後順序)
# 但由於market_size_inner_usage並非放在Data class中的__init__部分
# 因此在python編譯時，似乎還不會處理__init__以外的function
# 這些function在物件建立instance並呼叫對應function時才會被activated
# 所以不會有還沒宣告market_size_inner_usage的狀況
# 如此一來在以已有Data class下，最後才產生market_size_inner_usage
# 就可以發現其實存有先後關係因為market_size_inner_usage在建立時也只使用到__init__的部分
# (有測試過若把market_size_inner_usage放在__init__中就會出現未定義變數的問題，
# 由此可見在定義物件時會先編譯__init__的部分)

# 須注意在儲存市值檔案之csv檔時需要命名為「總市值(億)_已格式化」
# 否則在下面讀取之料時會有問題
market_size_inner_usage = Data('market_size', data_frequency='day')


# Note：
# 發現到actual_trade_time此變數其實也是寫在Data class宣告之後
# 但在Data class中的__init__部分其實已經使用到此變數，也沒有出現未定義的error
# 因此變數宣告上似乎不見得是前後順序，也有可能是宣告class此類特殊狀況時會有前後順序

# Reference:
# How do I sort a dictionary by value?
# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

# Filtering a list based on a list of booleans
# https://stackoverflow.com/questions/18665873/filtering-a-list-based-on-a-list-of-booleans

# Convert a pandas “Series of pair arrays” to a “two-column DataFrame”?
# https://stackoverflow.com/questions/29346512/convert-a-pandas-series-of-pair-arrays-to-a-two-column-dataframe