import pandas as pd



dfManual = {
    'Name': ['Tom', 'nick', 'krish', 'jack'],
    'Age': [20, 21, 19, 18],
    'salary': [2000, 3000, 4000, 5000],

}
print("Dataframe from manual data:")
df = pd.DataFrame(dfManual)
print(df)


print("\nDataframe from CSV file:")
dfcsv = pd.read_csv('H:\\Machine Learning\\lab1\\Data sets\\cars.csv')
print(dfcsv.head())


print("\nDataframe from Excel file:")
dfexcel = pd.read_excel('H:\\Machine Learning\\lab1\\Data sets\\extra.xlsx')
print(dfexcel.head())


print("\nDataframe from JSON file:")
dfjson = pd.read_json('H:\\Machine Learning\\lab1\\Data sets\\attendance.json')
print(dfjson.head())



print("\nDataframe from Text file:")
dftext = pd.read_csv('H:\\Machine Learning\\lab1\\Data sets\\data.txt', sep="\t")
print(dftext.head())

print("\nDataframe from Link:")
dflink = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv')
print(dflink.head())

print ("\nDataframe from Jason Link : ")
dfjsonlink = pd.read_json('https://jsonplaceholder.typicode.com/users')
print(dfjsonlink.head())
