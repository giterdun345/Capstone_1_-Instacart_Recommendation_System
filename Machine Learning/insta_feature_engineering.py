# -*- coding: utf-8 -*-
"""
@author: KETT
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings
import gc
# enable garbage collector to aid in memory 
gc.enable()
# eliminate future warnings
warnings.filterwarnings('ignore')


################################################################## Import Data
aisles = pd.read_csv("aisles.csv")
department = pd.read_csv("departments.csv")
orders = pd.read_csv("orders.csv")
prior = pd.read_csv("order_products__prior.csv")
train = pd.read_csv("order_products__train.csv")
products = pd.read_csv("products.csv")


# stock data frame for product and user_product features used later
stock = products.merge(department, on = 'department_id', how = 'left')
stock = products.merge(aisles, on = 'aisle_id', how = 'left')
stock.drop(['aisle', 'product_name'], axis = 1, inplace = True)


# prior_orders for user features, joins user_id and order_id 
prior_orders = prior.merge(orders, on = 'order_id', how = 'inner')

del products, aisles, department
gc.collect()

def memory_cruncher(df):
    """Reduces memory usage of data frame by converting the columns in 
    to smaller dtypes"""
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
        elif df[c].dtype == 'int64':
            df[c] = df[c].astype('int32')
        elif c == 'reordered':
            df[c] == df[c].astype('int8')
        else:
            pass
    return df


memory_cruncher(prior_orders);
memory_cruncher(stock);
print('Data has been imported and reduced...')

# # Predictor Variables of Market Basket Analysis
# <ol>
#   1. User prediction variables - behaviors of users <br>
#   2. Product prediction variables - info on products <br>
#   3. User/Product prediction variables - behavior towards a product
# </ol>

##################################################### User Predictor Variables
# total amount of orders by user 
user = prior_orders.groupby('user_id').order_number.max() \
        .to_frame('user_totals_u').reset_index()

# average number of orders per user
number_of_orders = prior_orders.groupby('user_id').order_number.mean() \
        .to_frame('number_of_orders_u').reset_index()

#avg number of days between orders
avg_days_between = prior_orders.groupby('user_id').days_since_prior_order.mean() \
        .to_frame('average_days_between_u').reset_index()

# average number for reordered products by user
reorder_ratio = prior_orders.groupby('user_id').reordered.mean() \
        .to_frame('reorder_ratio_u').reset_index()

# order day of the week with the most items 
most_dow = prior_orders.groupby('user_id').order_dow \
        .agg(lambda x:x.value_counts().index[0]) \
        .to_frame('most_dow_u').reset_index()

# order hour of the day with the most items
most_hour = prior_orders.groupby('user_id').order_hour_of_day \
        .agg(lambda x:x.value_counts().index[0]) \
        .to_frame('most_hour_u').reset_index()

# total amount of different items bought
specific_items = prior_orders.groupby('user_id').size() \
        .to_frame('specific_items_p').reset_index()

# creating the final df for user
user = user.merge(number_of_orders, on = 'user_id', how = 'left')
user = user.merge(avg_days_between, on = 'user_id', how = 'left')
user = user.merge(most_dow, on = 'user_id', how = 'left')
user = user.merge(most_hour, on = 'user_id', how = 'left')
user = user.merge(reorder_ratio, on = 'user_id', how = 'left')
user = user.merge(specific_items, on = 'user_id', how = 'left')

# reduce memory of user df
memory_cruncher(user)

del number_of_orders, avg_days_between, most_dow, most_hour,  \
    reorder_ratio, specific_items
gc.collect()
print('User df created...')
################################################## Product predictor variables 
# total amount of purchases per product
product = prior_orders.groupby('product_id').order_id.count() \
        .to_frame('total_purchased_p').reset_index()

# calculate the ratio of repurchase for each product
reorder_ratio = prior_orders.groupby('product_id').reordered.mean() \
        .to_frame('reorder_ratio_p').reset_index()

# calculate the mean for position that the product is added to the cart
addtocart = prior_orders.groupby('product_id').add_to_cart_order.mean() \
        .to_frame('cart_position_p').reset_index()

# Merge the product with reorder probability
product = product.merge(reorder_ratio, on='product_id', how='left')
# merge product with addtocart
product = product.merge(addtocart, on = 'product_id', how = 'left')

# reduce memory of product df
memory_cruncher(product);

del reorder_ratio, addtocart
gc.collect()
print('Product df created...')

############################################# User-Product predictor variables
# amount of times a user bought a specific product
user_product = prior_orders.groupby(by=['user_id', 'product_id']).order_id.count() \
        .to_frame('times_bought_up').reset_index()
# changing the dtype.
user_product['times_bought_up'] = user_product['times_bought_up'].astype(np.uint16)

# how many times a user purchased the product after purchasing it once,
#  same as above used to calculate ratio, support
times = prior_orders.groupby(['user_id','product_id']).order_id.count() \
        .to_frame('amt_bought').reset_index()

# total orders used for calculating range, support
total_orders = prior_orders.groupby('user_id')['order_number'].max() \
        .to_frame('total_orders').reset_index()

# finding when the user has bought a product the first time.
first_order_num = prior_orders.groupby(by=['user_id', 'product_id']) \
        ['order_number'].min().to_frame('first_order_num').reset_index()

# merging to make calculations on 
span = pd.merge(total_orders, first_order_num, on='user_id', how='right')

# calculating range, plus one for offset
span['range'] = span.total_orders - span.first_order_num + 1

# merging times df with the span to calculate reorder ratio
up_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')

#calculating the ratio.
up_ratio['reorder_ratio_up'] = up_ratio.amt_bought / up_ratio.range

# droppinf irrelevant columns
up_ratio.drop(
              ['amt_bought', 'total_orders', 'first_order_num', 'range'],
              axis=1,
              inplace=True
              )

del span, first_order_num, total_orders
gc.collect()

# merging up ratio to construct user-product df
user_product = user_product.merge(up_ratio,
                                  on=['user_id', 'product_id'], 
                                  how='left'
                                  )

# Reversing the order number for each product.
prior_orders['reversed_order'] = prior_orders.groupby(by=['user_id']) \
        ['order_number'].transform(max) - prior_orders.order_number + 1

# keeping only the first 5 orders from the order_number_back.
last_5 = prior_orders.loc[prior_orders.reversed_order <= 5]

# product bought by users in the last_five orders.
last_five = last_5.groupby(by=['user_id', 'product_id'])['order_id'].count() \
        .to_frame('last_five_up').reset_index()

# ratio of the products bought in the last_five orders.
last_five['ratio_last_five_up'] = last_five.last_five_up / 5.0

# merging this feature with uxp df.
user_product = user_product.merge(last_five,
                                  on=['user_id', 'product_id'],
                                  how='left'
                                  )

del last_five, last_5
gc.collect()

# filling the NAN values with 0 and reducing memory
user_product.fillna(0, inplace=True)
memory_cruncher(user_product)
print('User-Product df created...')

######################################### merge user, product, user-product df
# Merge user-product features with the user features
df1 = user_product.merge(user, on='user_id', how='left')
# Merging product features with df for a complete df 
# with all three categories of predictor variables
df1 = df1.merge(product, on='product_id', how='left')

del user, product, user_product, prior
gc.collect()
print('Dfs merged and deleted...')
################################################################ Preprocessing
# extracting training and test for identifying in df
future_orders = orders.loc[((orders.eval_set == 'train') | \
                            (orders.eval_set == 'test')),
                            [ 'user_id', 'eval_set', 'order_id']
                            ]

# test and train merged with df1 to allow for seperation
df1 = df1.merge(future_orders, on = 'user_id', how = 'left')

# seperation of training rows
df_train = df1[df1.eval_set == 'train']

# retrieves the reorder column from the train df
df_train = df_train.merge(train[['product_id', 
                                 'order_id',
                                 'reordered']],
                            on = ['product_id', 'order_id'],
                            how = 'left'
                            )

# order_id and eval set are no longer relevant
df_train.drop(['order_id', 'eval_set'], axis = 1, inplace = True)

# fill all NaN with zero, mostly reordered column
df_train.fillna(0, inplace = True)

# seperate test rows
df_test = df1[df1.eval_set == 'test']

# order+id and eval set no longer relevant to test df either
df_test.drop(['eval_set', 'order_id'], axis = 1, inplace = True)

# adding columns with aisle id and department id to be encoded 
df_train = df_train.merge(stock, on = 'product_id', how = 'left')
df_test = df_test.merge(stock, on = 'product_id', how = 'left')

del future_orders, df1, prior_orders, times, up_ratio
df_train.to_csv('df_train.csv', df_train.columns, index_label = 'user_id')
df_test.to_csv('df_test.csv', df_test.columns, index_label = 'user_id')
print('df_train and df_test has been created.')

df_train.info()

print('\n')
df_test.info()