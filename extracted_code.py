ie Eat Seach View Format Stax see Toots

Be omeex|s 2s «|x BalQan
a tmony

Yanport warnings

2 warnings. filterwarnings( ignore’)

4import streamlit as st

S import yan!

@ from yan}. loader import Safetoader

ist. set page config(layout-"wide”, page title

1)

if ‘user id’ not in st-session_ state
40) # username = ‘rajpurov’
ia] username = st-context headers. get

12} name = st.context-headers.get ("x

14) 4f ‘name’ not in st.session state
st.session_state['name’] = nam
username’ not in st-session_ state

st.session state[ ‘username’ ] - username

15
ie if
20 from initial_app import *
Biinit_app() ### Initialization of
23 import plotly-express as px

24 import plotly.graph_objects as go
25from plotly.subplots import make subplots
26 import pandas as pd

27 import logging

28 from connector.connectPropstore import

ds\from error handler import parse_snowflake_error
0 from sql_valid import query_validator

Sifrom datetime import date, timedelta,datetime
a2 import time

23 import json

44 import 05

35 from utility import *

Se-from custom components import custom toast

57 import snowflake. connector

Sel from chinou,client domain amport \iMDescriptor
Se\fron chinou.client .proxy amport Proxylin

Ee/AajpuroWAppData\Local TempiMiz0s\Re smotefiles\6678.

P Type here to search

g

tips

page_icon

initial _sidebar_state-“expanded”



opp omanpy

40 from chinou. util.acces rs
TLaccess amport ssokxception, get_sso token

si import requests
42 from requests.auth import HTTPBasicauth
43 from sql_queries import *
45 import zipfile
ae from streanlit theme inport st_thene
a7 theme ~ st_theme()
49 env-‘dev"
ols. print ("connecting to propstore.")
Siiproperty values ~ fetch properties(env)
sa Set up logging
U vogeine. basicconfig(Filenane:property values{ ‘chatbot log], level-logging.1Nf0, forest
Sou logging. basicconfig(filenane-’ ./chatbot-1og", x(asctime)s
$6
@st,title-"Governance Data Insights

Usertoken

level-logging-INFo, fornat

01 static variables

1 SIZE_LIMIT_MB - 190

62. MAX ROWS DISPLAY = 1000
63

os def local_css(file name)
66 with open(file_name,
Ist markdown(f’ <style>{f-read()}</style>

“e") as f
unsafe allow_html-True.

“7
6a

69)# local
70

a

ait Main app
7 def chatbot (owner_id)
74

% try:

cas(*/data/ntracedevpkg/dev/seripts/gov_insights/main-ce5")

Light”
‘acedevpke/dev/scri

v /light.css") as f
sread())</style>", unsafe.

allow html True:

if theme["base”]
with open("/data/ntr
ist markdown(f"<styler{t

wri203\RemoteFies\G678: 005

type here to search

%(asctine)s
%(levelname)s

X(levelnane)s

x(message)s'

Zi PM
AR) sara

:
|



a MobaTexésitor
Tile” fait Seaich View Format Syntax
Bone @x| 4 4|x &

yO eS
elif theme| "base ‘dark

with open(”/data/ntracedevpke/dev/scripts/gov_insights/dark.css") as f

st markdown(f"<style>(F.read()}</style>", unsafe allow html True’

except Exception as &
pass

header_columns = st.columns([

4 with header_columns[3].container(key-"head-col1")
if st.button("?2")
st.session state ight’ if st.session_state.theme == ‘dark’ else ‘dark’
st.rerun()

# Sidebar
with st.sidebar

st markdonn("chi style-‘color: red font-size: 4opx; font-weight: bold; margin-bottom: ©; padding-t scovernance data Insights</hi>”, unsafe allow html-True)

st.markdown 4 vata quality insights tool** ([User Guide] (https://confluence.nomura.con/TC8/confluence/display/Dataof fice/Nchat))")

Reset chat history

$F st.button("Clear chat History”, use container width-True)

st.session_state{ ‘chat _history
St-session state[ ‘last_question’] = None
st session state| "graph_but None
St.session_state[ “summary button”) = None
Stisession state[“file path"] - None

be i * 1 Rule and Except ‘DQ Rule Run "og Rule Process Run Log’ ‘Issue Tracker Data”, “Org Data(Operations)", “RMEP

option-st.selectbox("**Select an op
Exception Data", “Entity Data(cse

selected databa ption

St session state ‘selected

exceptd-pd.read_csv v/seripts/e mp

ICA Users rajpurov\AppData\Locel\Temp\Mxt20

PPE © type here to sear



iw MobaTextéditor
HP File Edit Search View Format Syntax Special Tools
momte x!) 44|ke Gleam ,

o rmanpy
‘end_date = end date placeholder.date input('*End Date**’,min_value-start_date,max_value-max value

#9 Database tables preview

st.markdown("## Data Preview
Af selected database —"Dg Rule and Except
with st.popover(‘#i## Exception Data
st.write(exceptd)
with st-popover # Rule Met
st.write(ruled
elif selected database-—"D@ Rule Run Log
with st.popover('#### Rule Run 0:
st write(rule_run_log’
With st.popover(‘#### Rule Metan:
st.write(ruled
elif option--"09 Rule Process Ri
with st.popove # Rul
st.write(rule process run
elif selected database -- “Navigator
with st.popover(‘#### Nav
st.write (navigator
with st.popover(‘###e Rule
st write(ruled
with st.popover(‘#ii# Excepti
st.write(exceptd
elif selected database —-"org 0:
with st.popover: 56 0
st.write(csg)
with st.popover(‘i#### OBI Data
st.write(obi)
elif selected database
with st.popover(“### RMEP Exce

Metapat

RMEP Exception Data’

st.write(rmep’
elif selected databa
with st-popover("##i E
st.write(entity
elif selected database
with st.popover(‘#
st.write(issue

elif selected database

{eAUsera\rajpurov\AppData\Local\Temp\Mxt203\RemoteFiles

Fim © type here to search

entity
ntity

Issue Tracker

Exception Cleaning

166781 005

ZIZPM |
9) spans



im Nobatextéditor
le Edit Search View Format’ Syntax Special Too's
[ a *|X RRQ QB
°
ruled-pd.read_csv("/data/ntracedevpkg
issue-pd-read csv( /data/ntracedevpkg/dev/scripts/gov_insights/sam
process. run-pd-read_csv("*/data/ntracedevpkg/dev/scripts/gov_insights
navigator - pd.read_csv(*/data/ntracedevpke/dev
ntrace = pd.read_csv("/data/ntracedevpke/dev/scripts/gov_insights/sample/ntrace. csv
pd.read_csv("/data/ntracedevpkg/dev/scripts/gov_insights/sample/jira.csv”
pd.read_csv("/data/ntracedevpke/dev/scripts/gov_insights/sample/csg.csv

pd.read_csv("/data/ntracedevpke/dev/scripts/gov_insights/sample/obi .csv”
insights/sample/rmep_exception_data.csv")

‘dev/scripts/gov_insights/sample/rule_metadata.csv"
le/Issuedata.csv
mple/rule_process_run_log.csv")

/scripts/gov_insights/sample/navigator.csv")

s/gov_insights/sample/entity data.csv")
s/sanple/rule_run_log.csv")

pmep-pd.read_esv("/data/ntracedevpke/dev/scri
entity-pd.read_csv("/data/ntracedevpkg/dev/scrip
rule run_log-pd.read_csv("/data/ntracedevpke/dev/scripts/gov_insigh

#a# Handling CoB data range
start_date - None
end_date - None
date columns - st.columns({1,
£ (option--"09 Rule and Exception Data") or (option
(option--"aMEP Exception Data”) or (option--"Entity Data(cse)"
with date columns[
start date placeholder - st.em
with date columns[1
end date placeholder - st.empty

‘09 Rule Run Log") or (option=="g Rule Process Run Log") er (option or (option--"org Data(operations)")or (

today - datetime.now()
min date - datetime(today.year-1,1,1
s*start pate**',min_value-min_date,max_value-datetime. today

start date — start_date placeholder. date_input

4f option —- “RMEP Exception Data
max value - start_datest imedelta(days-180)

else:
max value - start_datertimedelta(days-60,

if max value > date. today,
max value - datetime. today(

DOS Python d
22 PM
B® spsnms

P. Type here to search



with st.popover
st write(ruled
elif selected database Traceability
with st.popover('# raceability Data
st write(ntrace
elif selected database -—"21R/
with st.popover(‘#aea 31
st write jira
else
st write(“Please

Category Managenent
stmarkdown('## Cate ent
fcategory_columns = st.coluans((2, 1])
with category_colunns{o]
new category - st.chat_input("Enter your category
with category colums(1]
# if st.button(“Create catego
if new category
res ~ add category (new categ
af res
custom toast(F"{new cat
else
custom toast("Failed

time. sleep(1

categories ~ get categorie
with st-expander(label expanded-False
for i, cat in enunerate(categoric
cat_columns = st.columns((3, 1
with cat_columns(o
st write(c
with cat_colums(1
AF st.button("Renove", keynf
res = remove category (cat

FEE © Iypehereto search

yrsp2s



i MobaTextéditor
lg Edit Search View Format Syntax Special Too

+ e|x Bole

rmaney
af res
Custom toast (“category deleted", “success
else
‘custom toast("Failed to delete category”, “error”

time. sleep(1
st.rerun(

save category - st.selectbox("**s

ve Question:**", categories:
*name’ in st.session_state
st.markdown(f"<div class="welcom Leome back, <span class=‘username'>{st.session_state[ ‘name"]}</span></div>", unsafe allow html-True:

### Show chat history
# height ~ streamlit_js_eval(js_expressions='screen-height)
# container ht = int(height*.7)

# with st.container(height=container_ht, key="chat-container”)

for entry in st.session_state[’chat_history
with st.chat_message(name="user®, avatar-"user
st.urite(entry[ ‘question
with st-chat_message(name-"assistant”, avatar-"assistant
st.code(entry{"sql_query’], language-"sql
st-info("**Limited preview. Download complete dataset below
st.write(entry[ ‘result’ )
if entry| ‘chart
st.plotly chart(entry[ chart"), use_container_width-true, key-f"(uuid.uuidt()

4f entry[ ‘summary
st.info(entry| summary

fay All placeholders
user input placeholder — st.empty;

sql_response_placeholder - st.empty
data_response placeholder = st.
error placeholder = st-container
graph placeholder - st.empty
Summary placeholder = st-empty()

jefs\Vajpurow\AppData\Local\ Temp \Mxt203\Remoteles\66781 005

P. Iype here to search orm © Se WB



ia MobaTextEaitor

Tile Edit Search View Format Syntax Special To

form submit - st.empty:

#4 Instantiate buttons
button column = st.columns((1,1,1,1,2,2,2

HH Handling cob
cob = None
$F selected database--"0g rule and Exception Data” or selected database--"09 Rule Run Log” or selected database--"09 Rule Process Run Log” or selected database "Navigator”
selected database- “org Data(Operations)” or selected database--"RMEP Exception Data” or selected database-—"entity Data(csg)
cob = True
else
cob - False

#44 add disabled feature
#8 Upvote response
with button_colum[o]
if st.session_state["file path"] is not None:
4f st.button(”:thumbsup:”, use container width-True, help="Upyote
4f st-session_state.get(‘last_question’)
if update feedback("upvote’, st.session_state| ‘last _question‘ })
Custom toast("Thank you for your feedback!", “success”
else
custom _toast

Ree ans

ailed to update feedback. Please try again
else
custom toast ("No recent question to up warning’

time.sleep(1)
st.rerun()

### Downvote response
with button column{ 1]
if st.session_state|“file path” | is not None
Af st.button(™:thumbsdown:", use container width-True, help="Downvote
Af st,session state.get (‘last _question
4f update_feedback( ' downvot« session_state[ ‘last_question’ |
custom toast ("Your feedback will help us impro\
else
custom toast("Failed to update fee

[eAUsers\rajpurovsAppData\Local\TempNot202\Remotehiles\G678:005 Python

PD ype hereto search



in Mobatesttator
Ble Eat Search View Format Syntax
BOBS @ x| s elk &

Omen
time, sleep(1

### To save last question
with button coluan(2]
4f st.session_state["File path") is not None:
4f st.button(":page facing_up:", use cont
save_question ~ st.session state
sql_query = st.session state| ‘last_sql
res - add_question(save category, Save question, sql_query, selected database

ainer width-True, help: "sav

af res
custon_toast(F"Question saved to
else
custom toast(“Failed to save

ave_category)

time. sleep(1
st rerun!

#8 Working on download button
with button_column{ 3)
a2 Af st.session state|
20 File path - st.session_state[ “file path
Af o8.path-exists(File path.
with open(file path, "rb

4f st-download_button!

file_path"| is not None

as file

{Users ra) puroW\AppData\Local\Temp\Mxt203\Remotetiles\6678

Eis 2 type here to search

data file,

file _nane-os. path. basename

file path
mime-" application/zip

use container width-True
help-"ownload zip file

a print (f"File
st. session state
tt 1 can remove file using
# os. cemove( File path)

00S Python

cob

25 PM
32672005



File Edit "Search View Format Syntax Special To
Brome ex|2 24 4|xe

a)

## Generate Graph
with button column[5)
Af st.session state["graph_button”] is not None
Af st-button("Graph”, use container width-True:
aresult_df = pd.read_csv(st.session state. last df)
result df - pd.read_csv(st.session state. last_df, compression-‘zip’)
user_input = st.session state. last question

rs = graph(user_input, result_df, graph placeholder

Af rs
st-session_state["graph_button") ~ None
st-rerun(

else
custom toast (“sorry! Couldn't generate graph for this data”, “error”:

time. sleep(1

st.rerun(

#8 Generate summary

with button_column[6)
Af st.session_state| "summary button”) 4s not None and selected databa

Af st.button("summary", use container width-True.
result df = pd.read_csv(st.session state. last_df)
result df - pd.read_csv(st.session state. last_df, compression-"zip')

user input = st.session state. last_question

Fs — generate summary(user_input, result df, summary placeholder

if rs
st-session_state|"summary button
st rerun

else
custom toast

“sorry! Couldn't generate
time.sleep(1)
JE\Usera\ralpuron\AppData\Local\Temp\Mxt203\Remotefiles\6678 DOS Python

PPM © type here to search



