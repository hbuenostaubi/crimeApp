
import streamlit as st
import numpy as np, pandas as pd, joblib, altair as alt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def main():
    page = st.sidebar.selectbox("Choose a page for a model", ["Homepage", "SVM", "Neural Network", "Naive Bayes"])
    model = joblib.load('./Data/SVM_19_model')
    if page == "Homepage":
        st.header("Chicago Crime")


    elif page == "SVM":
        st.title("Data Exploration")

        df = load_data("SVM")
        st.header("Support Vector Machines")
        # st.write(df)
        visualize_data(df)

        option_town = st.selectbox('Select town:', ("North Lawndale", "Jefferson Park", "Chicago Lawn", "Irving Park", "Ashburn", "Austin", "Avalon Park", "East Garfield Park", "Oakland", "Roseland", "Englewood", "Humboldt Park", "Brighton Park", "Washington Park", "Belmont Cragin", "Avondale", "Greater Grand Crossing", "South Shore", "West Pullman", "Logan Square", "West Englewood",
                "Washington Heights", "Hermosa", "Near West Side", "Lincoln Park", "Hyde Park", "North Park", "West Lawn", "West      Town", "Near North Side", "Hegewisch", "South Chicago", "Calumet Heights", "Chatham", "Portage Park", "Auburn Gresham", "Woodlawn", "Albany Park", "South Deering", "South Lawndale", "West Garfield Park", "Lower West Side", "West Elsdon", "Uptown",
                "Gage Park", "West Ridge", "Fuller Park", "Riverdale", "Burnside", "Edgewater", "Lake View", "Lincoln Square", "East Side", "Mount Greenwood", "McKinley Park", "Loop", "Garfield Ridge", "Rogers Park", "Dunning", "Bridgeport", "Ohare", "Norwood Park", "Morgan Park", "Edison Park", "Forest Glen", "Grand Boulevard", "Clearing", "Near South Side", "North Center",
                "Kenwood", "Pullman", "Archer Heights,", "Beverly", "Douglas", "Montclare", "Armour Square"))
        option_year = st.selectbox('Select year', (2017, 2018,2019))
        option_month = st.selectbox('Select month', ('Jan', 'Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        option_time = st.selectbox('Select time of day', ('Evening', 'Night','Morning', 'Afternoon'))
        month_dic = {'Jan':1, 'Feb':2,'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        # st.write('You selected:', option, option2, option3, option4)
        df_train = load_training()
        model = getModel('SVM')
        if st.button('submit'):
            Xnew = predictDummy(month_dic[option_month], option_town, option_time, option_year, df_train)  #valMonth, valCommy, valTimeDay, valYear, df
            res_svm = getMaxCats(Xnew, model)
            st.table(res_svm)

    elif page == "Neural Network":
        df = load_data("NN")
        st.header("Neural Network")
        visualize_data(df)
        option_town = st.selectbox('Select town:', ("North Lawndale", "Jefferson Park", "Chicago Lawn", "Irving Park", "Ashburn", "Austin", "Avalon Park", "East Garfield Park", "Oakland", "Roseland", "Englewood", "Humboldt Park", "Brighton Park", "Washington Park", "Belmont Cragin", "Avondale", "Greater Grand Crossing", "South Shore", "West Pullman", "Logan Square", "West Englewood",
                "Washington Heights", "Hermosa", "Near West Side", "Lincoln Park", "Hyde Park", "North Park", "West Lawn", "West      Town", "Near North Side", "Hegewisch", "South Chicago", "Calumet Heights", "Chatham", "Portage Park", "Auburn Gresham", "Woodlawn", "Albany Park", "South Deering", "South Lawndale", "West Garfield Park", "Lower West Side", "West Elsdon", "Uptown",
                "Gage Park", "West Ridge", "Fuller Park", "Riverdale", "Burnside", "Edgewater", "Lake View", "Lincoln Square", "East Side", "Mount Greenwood", "McKinley Park", "Loop", "Garfield Ridge", "Rogers Park", "Dunning", "Bridgeport", "Ohare", "Norwood Park", "Morgan Park", "Edison Park", "Forest Glen", "Grand Boulevard", "Clearing", "Near South Side", "North Center",
                "Kenwood", "Pullman", "Archer Heights,", "Beverly", "Douglas", "Montclare", "Armour Square"))
        option_year = st.selectbox('Select year', (2017, 2018,2019))
        option_month = st.selectbox('Select month', ('Jan', 'Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        option_time = st.selectbox('Select time of day', ('Evening', 'Night','Morning', 'Afternoon'))
        month_dic = {'Jan':1, 'Feb':2,'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        model_NN = getModel('NN')
        if st.button('submit'):
            x_new = getPrimNN(option_year,option_town,month_dic[option_month],option_time ,load_training())
            yhat=np.ndarray.tolist(model_NN.predict(x_new, batch_size=32, verbose=0))
            res_NN = getMaxPrim(yhat, load_yvals())
            st.table(res_NN)
    elif page == "Naive Bayes":
        df = load_data("NB")
        st.header("Naive Bayes")
        # st.write(df)
        visualize_data(df)

        df_NB = load_training()
        likely = getLikelihood(df_NB)
        option_town = st.selectbox('Select town:', ("North Lawndale", "Jefferson Park", "Chicago Lawn", "Irving Park", "Ashburn", "Austin", "Avalon Park", "East Garfield Park", "Oakland", "Roseland", "Englewood", "Humboldt Park", "Brighton Park", "Washington Park", "Belmont Cragin", "Avondale", "Greater Grand Crossing", "South Shore", "West Pullman", "Logan Square", "West Englewood",
                "Washington Heights", "Hermosa", "Near West Side", "Lincoln Park", "Hyde Park", "North Park", "West Lawn", "West      Town", "Near North Side", "Hegewisch", "South Chicago", "Calumet Heights", "Chatham", "Portage Park", "Auburn Gresham", "Woodlawn", "Albany Park", "South Deering", "South Lawndale", "West Garfield Park", "Lower West Side", "West Elsdon", "Uptown",
                "Gage Park", "West Ridge", "Fuller Park", "Riverdale", "Burnside", "Edgewater", "Lake View", "Lincoln Square", "East Side", "Mount Greenwood", "McKinley Park", "Loop", "Garfield Ridge", "Rogers Park", "Dunning", "Bridgeport", "Ohare", "Norwood Park", "Morgan Park", "Edison Park", "Forest Glen", "Grand Boulevard", "Clearing", "Near South Side", "North Center",
                "Kenwood", "Pullman", "Archer Heights,", "Beverly", "Douglas", "Montclare", "Armour Square"))
        option_year = st.selectbox('Select year', (2017, 2018,2019))
        option_month = st.selectbox('Select month', ('Jan', 'Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        option_time = st.selectbox('Select time of day', ('Evening', 'Night','Morning', 'Afternoon'))
        month_dic = {'Jan':1, 'Feb':2,'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

        if st.button('submit'):
            res_NB = getPrimary5(option_year,option_town, month_dic[option_month], option_time, likely, df_NB)
            st.table(res_NB)

@st.cache
def load_data(key_word):
    if(key_word == "SVM"):
        # df = pd.read_csv('[prior in masterThesis Folder under Data directory]/Data/new_freq_SVM.csv').sort_values(by=['count'])
        df = pd.read_csv('./Data/new_freq_SVM.csv')
    elif key_word =="NN":
        df = pd.read_csv('./Data/new_freq_NN.csv')
    elif key_word =="NB":
        df = pd.read_csv('./Data/new_freq_NB.csv')
    return df.nlargest(5, 'count')


def load_training():
    return pd.read_pickle('./Data/training19_data')

def visualize_data(df):
    fig, ax = plt.subplots(figsize=(4, 6))
    y = df['count'].tolist()
    x = df['primary_type'].tolist()
    bars = ax.barh(x,y,.5)
    for bar in bars:
        width = bar.get_width()
        label_y_pos = bar.get_y() + bar.get_height() / 6
        ax.text(width, label_y_pos, s=f'{width}', va='center')

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(5)
    for bar in bars[::2]:
        bar.set_color('r')
    plt.title('Frequency Distribution')
    # plt.xlabel('Starting hp')
    # plt.ylabel('Champion name')
    plt.xlim([0, 80000])
    ax.bar(y,x)
    st.pyplot(plt)

def getModel(key_word):
    if(key_word == "SVM"):
        return joblib.load('./Data/SVM_19_model')
    elif(key_word == 'NN'):
        return load_model('./Data/my_model')

def predictDummy(valMonth, valCommy, valTimeDay, valYear, df):      ### df is named globally
    df_1 = pd.DataFrame(index=range(0,1), columns = df['month_num'].unique()).fillna(0)
    df_2 = pd.DataFrame(index=range(0,1), columns = df['community_name'].unique()).fillna(0)
    df_3 = pd.DataFrame(index=range(0,1), columns = df['time_category'].unique()).fillna(0)
    df_4 = pd.DataFrame(index=range(0,1), columns = df['year'].unique()).fillna(0)
    # df_4 = pd.DataFrame({'year':[valYear]})
    df_1[valMonth][0]+=1
    df_2[valCommy][0]+=1
    df_3[valTimeDay][0]+=1
    df_4[valYear][0]+=1
    return pd.concat([df_4, df_1, df_2, df_3], axis=1).values

def getMaxCats(XNew, model):
    ynew = model.predict_proba(XNew)
    ycat = model.classes_

    y_df = pd.DataFrame(index=range(0, 5), columns = ['y','cat'])
    pd.set_option('mode.chained_assignment', None)
    for i in range(0,5):
        y_df['y'][i]=ynew[0][i]
        y_df['cat'][i]=ycat[i]
    minVal=y_df['y'].min()
    y_i=np.where(y_df['y']==minVal)[0][0]
    for i in range(5,ynew.shape[1]):
        if(ynew[0][i]>minVal):
            y_df['cat'][y_i] = ycat[i]
            y_df['y'][y_i] = ynew[0][i]
            minVal=y_df['y'].min()
            y_i=np.where(y_df['y']==minVal)[0][0]
    y_df = y_df.sort_values(by = ['y'], ascending = False, ignore_index=True)
    y_df['y']= y_df['y'].apply(lambda x: "{:.2%}".format(x))
    col_names = ['cat', 'y']
    y_df = y_df.reindex(columns=col_names)
    y_df = y_df.rename({'cat':'Category', 'y': 'Max Probabilities'}, axis=1)
    return y_df

# This section is for NaiveBayes
def getPrimary5(year,town, month, timeType, likelihood, df):
    df_community = df[['community_name','community_area']].drop_duplicates()
    if(type(town)== str):
        town=df_community[df_community['community_name']==town]['community_area'].values[0]
    pLst=list(df['primary_type'].unique())
    hLst=['primary_type', 'time_category','community_area', 'month_num', 'year']
    df=df[hLst]
    dayTime=['Afternoon','Evening', 'Morning', 'Night']
    ### Naive Bayes
    j = dayTime.index(timeType)
    prior= df.groupby(hLst[0]).size().div(len(df))  #count()['Age']/len(data)
    numMaxMin=0
    primMax=''
    df_max = pd.DataFrame(data = {'PrimaryType':[],'probMax': []})
    for i in pLst:   ## primary list gets assigned below
        primary=i
        timeType=j   ## time of day in categorical value
        temp=likelihood[hLst[1]][primary].get(j,0)*likelihood[hLst[2]][primary].get(town,0)*likelihood[hLst[3]][primary].get(month,0)*likelihood[hLst[4]][primary].get(year,0)*prior[primary]
        if(df_max.shape[0]<5):
            df_max = df_max.append({'PrimaryType': primary,'probMax': temp}, ignore_index=True)
        if(temp>numMaxMin):
            df_max = df_max.append({'PrimaryType': primary,'probMax': temp}, ignore_index=True)
            numMaxMin=temp
            if(df_max.shape[0]>5):
                numMaxMin=getMinDF(df_max)
                df_max = df_max.drop(numMaxMin)
                df_max = df_max.sort_values(by=['probMax']).reset_index(drop=True)
            numMaxMin = getMinDF(df_max)
        df_max = df_max.drop_duplicates()
    df_max['probMax']=df_max['probMax'].apply(lambda x: "{:.4%}".format(x))
    df_max = df_max.rename({'PrimaryType':'Category', 'probMax': 'Max Probabilities'}, axis=1)
    return df_max.sort_values(by=['Max Probabilities'], ascending=False).reset_index(drop=True)

def getLikelihood(df):
    hLst=['primary_type', 'time_category','community_area', 'month_num', 'year']
    df=df[hLst]
    prior= df.groupby(hLst[0]).size().div(len(df))  #count()['Age']/len(data)
    likelihood = {}
    for i in range(len(hLst)):
        likelihood[hLst[i]]=df.groupby([hLst[0],hLst[i]]).size().div(len(df)).div(prior)
    return likelihood

def getMinDF(df):
    return df['probMax'].idxmin(axis=0, skipna=True)

####### Neural Network Functions
def getPrimNN(year,town,month, timeType ,df):
    df1 = getDum1(timeType, 'time_category', df)
    df2 = getDum1(month, 'month_num', df )
    df3 = getDum1(town, 'community_name', df)
    df4 = getDum1(year, 'year', df)
    x = pd.concat([df4, df2, df3, df1], axis=1).values
    return np.ndarray.tolist(x)

def getMaxPrim(lst1, lst2):
    y_df = pd.DataFrame(index=range(0, 5), columns = ['y','cat'])
    pd.set_option('mode.chained_assignment', None)
    for i in range(0,5):
        y_df['y'][i]=lst1[0][i]
        y_df['cat'][i]=lst2[i]
    minVal=y_df['y'].min()
    y_i=np.where(y_df['y']==minVal)[0][0]
    for i in range(5,len(lst1[0])):
        if(lst1[0][i]>minVal):
            y_df['cat'][y_i] = lst2[i]
            y_df['y'][y_i] = lst1[0][i]
            minVal=y_df['y'].min()
            y_i=np.where(y_df['y']==minVal)[0][0]
    y_df = y_df.sort_values(by = ['y'], ascending = False, ignore_index=True)
    return y_df
def getDum1(str1, cat, df):  ###get def, transpose by cat and delete
    df_new = pd.DataFrame(index=[0], columns = df[cat].unique()).fillna(0)
    df_new[str1][0]=1
    return df_new

def load_yvals():
    return np.load('./Data/y_vals_NN.npy', allow_pickle=True)

if __name__ == "__main__":
    main()
