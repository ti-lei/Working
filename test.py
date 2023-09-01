# 自製模組套件
from preprocess import Data
from backtest import Backtest
from analysis import Analysis
# 外部套件
import os
from operators import lag,lag_series_for_change_in_a_row,lag_series_more_careful

def main():
    data_name = 'EPS_MEAN'

    # 建立Data物件
    week_growth = Data(
        data_name=data_name,
        data_frequency='week',
        market_size_threshold=10
    )
    def growth_filter(series):
        '''
        營收YoY負轉正
        '''
        return (
            lag_series_more_careful(series,5,)
        )
    # week_growth.mutate_data(
    #     None, 
    #     rebalance_frequency='week',
    #     date_order_ascending=False,
    #     measure_on_nth_date=1
    # )

    week_growth.add_filter(growth_filter)

    
    signal_list = [
        week_growth
    ]

    backtesting = Backtest(
        signal_list,
        start_date='2010/1/1',
        end_date='2023/7/30'
    )
    backtesting.run()
    

    backtesting_result = Analysis(backtesting, tax_rate=0, fee_rate=0)
    backtesting_result.run()
if __name__=="__main__":
    main()
