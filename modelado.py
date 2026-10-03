import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df=pd.read_csv("data/data.csv",sep=";")
df["Desercion"]=(df["Target"]=="Dropout").astype(int)
for tag,sem in [("1S","1st"),("2S","2nd")]:
    e=f"Curricular units {sem} sem (enrolled)"; a=f"Curricular units {sem} sem (approved)"
    df[f"Tasa_aprobacion_{tag}"]=np.where(df[e]>0,df[a]/df[e],0).clip(0,1)
df["Cambio_calificacion"]=df["Curricular units 2nd sem (grade)"]-df["Curricular units 1st sem (grade)"]
df["Riesgo_financiero"]=((df["Debtor"]==1)|(df["Tuition fees up to date"]==0)).astype(int)
y=df.pop("Desercion"); df=df.drop(columns=["Target"]); X=df
cat=["Marital status","Application mode","Course","Daytime/evening attendance","Previous qualification","Nacionality","Mother's qualification","Father's qualification","Mother's occupation","Father's occupation","Displaced","Educational special needs","Debtor","Tuition fees up to date","Gender","Scholarship holder","International","Riesgo_financiero"]
num=[c for c in X.columns if c not in cat]
def prep(): return ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
models={"Regresión logística":LogisticRegression(C=1,solver="liblinear",max_iter=1000,class_weight="balanced",random_state=42),"Árbol de decisión":DecisionTreeClassifier(max_depth=6,min_samples_split=20,min_samples_leaf=10,class_weight="balanced",random_state=42),"Random Forest":RandomForestClassifier(n_estimators=300,min_samples_split=4,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",random_state=42,n_jobs=-1)}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scoring={"Exactitud":"accuracy","Precisión":"precision","Sensibilidad":"recall","F1":"f1","ROC_AUC":"roc_auc"}
rows=[]
for name,m in models.items():
    p=Pipeline([("prep",prep()),("model",m)])
    s=cross_validate(p,Xtr,ytr,cv=cv,scoring=scoring,n_jobs=-1)
    row={"Modelo":name}
    for met in scoring: row[f"{met}_media"]=s[f"test_{met}"].mean()*100; row[f"{met}_DE"]=s[f"test_{met}"].std()*100
    rows.append(row)
pd.DataFrame(rows).round(2).to_csv("results/validacion_cruzada_5_folds.csv",index=False)
print(pd.DataFrame(rows).round(2))
