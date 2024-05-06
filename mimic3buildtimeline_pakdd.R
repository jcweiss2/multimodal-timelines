### This extraction code is for MIMIC III v 1.4
### Given the mimic directory ('mimicdir'), a file with desired HADM_ID in column 'pt',
### produce a 4-column timeline (pt, time, event value), and save it into the folder 'prefix'.

### Note: the code loads the chartevents file which requires large RAM (tested to run on 128GB + swap).
### When run with the i2b2 matched cohort (PAKDD 2024), we get a timeline file with 1647564 lines.


### Load libraries
library(tidyverse)
library(data.table)

### Set path and directory locations
setwd("<path to your working directory>")
mimicdir = "<path to your mimic 3 v1.4 dir>"
match_file = "matched.csv"
nameExtension = "pakdd2024"
prefix = paste0("r/extractions/",nameExtension,"/")

### Create 'prefix' directory
if(!file.exists(prefix)) dir.create(prefix, recursive = T)

### Helper function(s)
event_filter = function(data, subset, numberConditions = 2) {
  if(numberConditions == 1)
    return(data %>% filter(SUBJECT_ID %in% (subset$SUBJECT_ID)))
  return(data %>% filter(SUBJECT_ID %in% (subset$SUBJECT_ID) & HADM_ID %in% subset$HADM_ID))
}
rn = function(tuple, convertTimeToLong=T, formatting="%Y-%m-%d %H:%M:%S", formatting2="%Y-%m-%d") {
  # 'rn' renames the 4 column file and does time formatting
  names(tuple) = c("pt","t","event", "value")[1:length(tuple)]
  if(convertTimeToLong) {
    result = tuple %>% mutate(t1=as.POSIXct(strptime(t, format=formatting, tz="UTC"))%>%unclass()) %>%
      mutate(t = case_when(is.na(t1) ~ as.POSIXct(strptime(t, format=formatting2, tz="UTC")) %>% unclass(),
                                             T ~ t1)) %>% select(-t1)
    return(result)
  }
  tuple
}
rn2 = function(double, newcol="feature") {
  # 'rn2' renames columns with ("pt" and 'double')
  names(double) = c("pt",newcol)
  double
}

### Load requisite files from 'mimicdir'

### Inclusion: all admissions ('adm') to be associated with patients in 'match_file'
adm = read_csv(paste0(mimicdir,"ADMISSIONS.csv")) %>% tbl_df()

### Load all descriptive tables
dlist = list()
dfiles = list.files(mimicdir) %>% tbl_df() %>% filter(startsWith(value,"D_")) %>% t() %>% c()
for(i in 1:length(dfiles)) {
  dlist[[i]] = list(name=dfiles[i], data=read_csv(paste0(mimicdir,dfiles[i])) %>% tbl_df())
}

### Load remaining csv files 1 by 1 and collect timestamps
pfiles = list.files(mimicdir) %>% tbl_df() %>% filter(!startsWith(value,"D_") & endsWith(value,".csv")) %>% t() %>% c()

diagnoses = read_csv(paste0(mimicdir,"DIAGNOSES_ICD.csv")) %>% tbl_df()
desc_diagnoses = read_csv(paste0(mimicdir,"D_ICD_DIAGNOSES.csv")) %>% tbl_df()
ds = diagnoses %>% inner_join(desc_diagnoses %>% select(ICD9_CODE,LONG_TITLE), by="ICD9_CODE")

cohort = 
  fread(match_file) %>% as_tibble() %>% select(HADM_ID=hadm_id) %>% distinct() %>%
  left_join(
    adm %>% select(SUBJECT_ID, HADM_ID),
    by=c("HADM_ID")
  ) %>% select(SUBJECT_ID, HADM_ID)

### Initialize empty tables
tl = data.frame(pt=vector("numeric"), t=vector("numeric"), event=vector("character"), value=vector("character")) %>% tbl_df() %>%
  mutate(value = as.character(value))
constants = data.frame(pt=vector("numeric")) %>% tbl_df()

# Pull triples as you go (since you will {load, extract, and remove} the files sequentially)
# Pull descriptive tables as you go
adm_cohort = adm %>% event_filter(cohort)
adm_cohort = adm_cohort %>%
  mutate(ADMISSION_LOCATION=paste0("ADMISSION_LOCATION:",ADMISSION_LOCATION)) %>%
  mutate(ADMISSION_TYPE=paste0("ADMISSION_TYPE:",ADMISSION_TYPE)) %>%
  mutate(INSURANCE=paste0("INSURANCE:",INSURANCE)) %>%
  mutate(DISCHARGE_LOCATION=paste0("DISCHARGE_LOCATION:",DISCHARGE_LOCATION)) %>%
  mutate(MARITAL_STATUS=paste0("MARITAL_STATUS:",MARITAL_STATUS)) %>%
  mutate(DIAGNOSIS=paste0("DISCHARGE_DIAGNOSIS:",DIAGNOSIS))
tl = tl %>%
  bind_rows(adm_cohort %>% select(SUBJECT_ID,ADMITTIME, ADMISSION_LOCATION) %>% rn()) %>%
  bind_rows(adm_cohort %>% select(SUBJECT_ID,ADMITTIME, ADMISSION_TYPE) %>% rn()) %>%
  bind_rows(adm_cohort %>% select(SUBJECT_ID,ADMITTIME, INSURANCE) %>% rn() ) %>%
  bind_rows(adm_cohort %>% select(SUBJECT_ID,DISCHTIME, DISCHARGE_LOCATION) %>% rn()) %>%
  bind_rows(adm_cohort %>% select(SUBJECT_ID,ADMITTIME, MARITAL_STATUS) %>% rn()) %>%
  bind_rows(adm_cohort %>% select(SUBJECT_ID,DISCHTIME, DIAGNOSIS) %>% rn()) %>%
  bind_rows(adm_cohort %>% select(SUBJECT_ID,DISCHTIME) %>% mutate(DISCHARGE="DISCHARGED") %>% rn()) %>%
  arrange(pt)
constants = constants %>%
  full_join(adm_cohort %>% select(SUBJECT_ID,ETHNICITY) %>% mutate(ETHNICITY = paste0("ETHNICITY:",ETHNICITY)) %>% 
              rn2("ETHNICITY"), by="pt") %>% distinct()

# attach gender and age
pts = read_csv(paste0(mimicdir,"PATIENTS.csv")) %>% tbl_df()
pts_cohort = pts %>% event_filter(cohort, 1)
pts_cohort = pts_cohort %>% mutate(GENDER=paste0("GENDER:",GENDER))
constants = constants %>%
  full_join(pts_cohort %>% select(SUBJECT_ID, GENDER) %>% rn2("GENDER"))
tl = tl %>%
  bind_rows(pts_cohort %>% select(SUBJECT_ID, DOB) %>% mutate(BIRTHEVENT="BIRTHED") %>%
              rn(formatting="%Y-%m-%d")) %>% arrange(pt)
tl = tl %>%
  bind_rows(pts_cohort %>% select(SUBJECT_ID, DOB) %>% inner_join(constants, by=c("SUBJECT_ID" = "pt")) %>% select(-GENDER) %>% rn(formatting="%Y-%m-%d"),
            pts_cohort %>% select(SUBJECT_ID, DOB) %>% inner_join(constants, by=c("SUBJECT_ID" = "pt")) %>% select(-ETHNICITY) %>% rn(formatting="%Y-%m-%d"))
tl = tl %>%
  bind_rows(pts_cohort %>% select(SUBJECT_ID, DOD) %>% mutate(DEATHEVENT="DECEASED") %>% filter(!is.na(DOD)) %>%
              rn(formatting="%Y-%m-%d")) %>% arrange(pt)
rm(pts,pts_cohort); gc()


# detemine who is in carevue (2001-2008) and who is in metavision (2008-2012+)
icustays = read_csv(paste0(mimicdir,"ICUSTAYS.csv")) %>% tbl_df()
icu_cohort = icustays %>% event_filter(cohort) %>% 
  select(SUBJECT_ID,HADM_ID, INTIME, DBSOURCE)
tl = tl %>% bind_rows(
  icu_cohort %>% select(-HADM_ID) %>% rn()
)

# codiagnoses
ds = ds %>% event_filter(cohort) %>% mutate(LONG_TITLE=paste0("CoDx:",LONG_TITLE))
tl = tl %>% bind_rows(
  ds %>% 
    left_join(adm_cohort %>% select(SUBJECT_ID, HADM_ID, DISCHTIME), by=c("SUBJECT_ID","HADM_ID")) %>%
    select(SUBJECT_ID, DISCHTIME, LONG_TITLE) %>% rn()
)
tl = tl %>% bind_rows(
  adm_cohort %>% filter(HOSPITAL_EXPIRE_FLAG==1) %>% select(SUBJECT_ID, DISCHTIME) %>% mutate("EXPIRED_IN_HOSPITAL") %>% rn()
)
rm(ds); gc()

# load prescriptions
prescriptions = read_csv(paste0(mimicdir,"PRESCRIPTIONS.csv")) %>% tbl_df()
prescriptions_cohort = prescriptions %>% event_filter(cohort)
tl = tl %>% bind_rows(
  prescriptions_cohort
 %>% mutate(DRUG=paste0("PRESCRIBED:",DRUG," via ",ROUTE, ":",DOSE_UNIT_RX)) %>%
    select(SUBJECT_ID, STARTDATE, DRUG, DOSE_VAL_RX) %>% 
    mutate(STARTDATE = STARTDATE + 60*60*24 - 1) %>%
    rn()
) %>% arrange(pt)
rm(prescriptions,prescriptions_cohort); gc()

# load lab tests
lab = read_csv(paste0(mimicdir,"LABEVENTS.csv")) %>% tbl_df()
dlab = read_csv(paste0(mimicdir,"D_LABITEMS.csv")) %>% tbl_df()
lab_cohort = lab %>% event_filter(cohort, numberConditions = 1) # outpatient labs do not have an HADM_ID (second condition)
lab_cohort = lab_cohort %>% left_join(dlab %>% select(LABEL,ITEMID, FLUID, CATEGORY), by="ITEMID") %>%
  mutate(LONGLABEL=paste0("LAB:",FLUID,":",CATEGORY,":",LABEL))
tl = tl %>% bind_rows(
  lab_cohort %>% select(SUBJECT_ID, CHARTTIME, LONGLABEL, VALUE) %>% rn()
)
rm(lab,dlab,lab_cohort); gc()


# load chartevents
# # here is a workaround if you are memory limited
# jumpsize = 1.7e8
# parts = data.frame(start = seq(from=1, by = jumpsize, to=330712484)) %>% tbl_df()
# parts = parts %>% mutate(end = lead(start)-1) %>%
#   rowwise() %>% do(ces = fread(paste0(mimicdir,"CHARTEVENTS.csv"), skip = .$start, nrows = jumpsize) %>% tbl_df() %>% filter(V2 %in% (<YOUR-SUBJECT_IDS-HERE>%>%t())))
# ce_cohort = unnest(parts)
# getNames = read_csv(paste0(mimicdir,"CHARTEVENTS.csv"), n_max = 2)
# names(ce_cohort) = names(getNames)
# write_csv(ce_cohort, paste0(mimicdir,"/extracts/CHARTEVENTS_cohort.csv"))
# ce = read_csv(paste0(mimicdir,"CHARTEVENTS.csv")) %>% tbl_df() # perhaps too big
# ce = read_csv(paste0(mimicdir,"/extracts/CHARTEVENTS_cohort.csv"),guess_max = 5e6) %>% tbl_df()
ce = fread(paste0(mimicdir,"CHARTEVENTS.csv")) %>% as_tibble()
# ce = ce %>% event_filter(<YOUR-SUBJECT_IDS-HERE>)
dce = read_csv(paste0(mimicdir,"D_ITEMS.csv")) %>% tbl_df()
ce_cohort = ce %>% filter(SUBJECT_ID %in% (cohort%>%t()))
ce_cohort = ce_cohort %>% left_join(dce %>% select(LABEL, ITEMID), by="ITEMID")
ce_cohort = ce_cohort %>% mutate(LONGLABEL=paste0("CHART:",LABEL,":",VALUEUOM))
tl = tl %>% bind_rows(
  ce_cohort %>% select(SUBJECT_ID, CHARTTIME, LONGLABEL, VALUE) %>% rn()
)
rm(ce,dce,ce_cohort); gc()

# icu timing; note that the 'value' column will keep the length of stay, so do not use it in your features for most tasks
icus = read_csv(paste0(mimicdir,"ICUSTAYS.csv")) %>% tbl_df()
icu_cohort = icus %>% event_filter(cohort)
tl = tl %>% bind_rows(
  icu_cohort %>% select(SUBJECT_ID, INTIME, LOS) %>% mutate(LABEL="inICU") %>%
    bind_rows(icu_cohort %>% select(SUBJECT_ID, OUTTIME, LOS) %>% rename(INTIME=OUTTIME) %>% mutate(LABEL="outICU") ) %>% 
    select(SUBJECT_ID, INTIME, LABEL, LOS) %>% mutate(LOS = as.character(LOS)) %>% rn()
)
rm(icus, icu_cohort)


# load microbiology tests/results
micro = read_csv(paste0(mimicdir,"MICROBIOLOGYEVENTS.csv")) %>% tbl_df()
micro_cohort = micro %>% event_filter(cohort)
micro_cohort = micro_cohort %>% select(SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, SPEC_TYPE_DESC, ORG_NAME) %>%
  mutate(CTIME = ifelse(is.na(CHARTTIME), yes = CHARTDATE + 60*60*24 - 1, no = CHARTTIME)) %>% # use end-of-day timestamp if not available
  mutate(LONGLABEL=paste0("MICROBIOLOGY:",SPEC_TYPE_DESC,":",ORG_NAME)) %>%
  select(SUBJECT_ID, HADM_ID, CTIME, LONGLABEL) %>% distinct() 
# not currently keeping sensitivity profiles of positive cultures but one could
tl = tl %>% bind_rows(
  micro_cohort %>% select(-HADM_ID) %>% (function(tuple) {names(tuple) = c("pt","t","event")[1:length(tuple)]; tuple})(.) %>% mutate(value = NA)
)
rm(micro, micro_cohort); gc()

# callout data
callout = read_csv(paste0(mimicdir,"CALLOUT.csv")) %>% tbl_df()
callout_cohort = callout %>% event_filter(cohort) %>% 
  select(SUBJECT_ID,HADM_ID, CURR_CAREUNIT, CALLOUT_SERVICE, CREATETIME, OUTCOMETIME, CALLOUT_OUTCOME) %>% arrange(SUBJECT_ID, CREATETIME)
tl = tl %>% bind_rows(
  callout_cohort %>% select(SUBJECT_ID, CREATETIME, CURR_CAREUNIT) %>% mutate(CURR_CAREUNIT = paste0("CALLOUT:DCSCHEDULED:",CURR_CAREUNIT)) %>% rn(),
  callout_cohort %>% select(SUBJECT_ID, CREATETIME, CALLOUT_SERVICE, CALLOUT_OUTCOME) %>% mutate(CALLOUT_SERVICE = paste0("CALLOUT:DCTO:",CALLOUT_SERVICE)) %>% rn()
)
rm(callout, callout_cohort); gc()

# CPTevents - not useful

# datetimevents: t/l/d's
datetimeevents = read_csv(paste0(mimicdir,"DATETIMEEVENTS.csv"), guess_max = 5e5) %>% tbl_df()
dce = read_csv(paste0(mimicdir,"D_ITEMS.csv")) %>% tbl_df()
dt_cohort = datetimeevents %>% event_filter(cohort) %>% 
  left_join(dce %>% filter(LINKSTO=="datetimeevents") %>% select(ITEMID, LABEL), by="ITEMID") %>%
  filter(!str_detect(pattern="^INV|Change", LABEL))
tl = tl %>% bind_rows(
  dt_cohort %>% select(SUBJECT_ID, VALUE, LABEL) %>% rn()
)
rm(datetimeevents, dce, dt_cohort); gc()

# DRG codes 
drgs = read_csv(paste0(mimicdir,"DRGCODES.csv"), guess_max = 5e5) %>% tbl_df()
drgs = drgs %>%
  inner_join(adm_cohort %>% select(SUBJECT_ID, HADM_ID, DISCHTIME), by=c("HADM_ID", "SUBJECT_ID")) 
tl = tl %>% bind_rows(
  drgs %>% select(SUBJECT_ID, DISCHTIME, DESCRIPTION, DRG_SEVERITY) %>% mutate(DRG_SEVERITY=DRG_SEVERITY %>% as.character()) %>% rn()
)

#INPUT events
inevents = read_csv(paste0(mimicdir,"INPUTEVENTS_CV.csv"), guess_max = 1e5) %>% tbl_df()
dce = read_csv(paste0(mimicdir,"D_ITEMS.csv")) %>% tbl_df()
in_cohort = inevents %>% event_filter(cohort) %>% 
  left_join(dce %>% select(ITEMID, LABEL), by="ITEMID") %>%
  mutate(LONGLABEL=paste0("INPUT:",ORIGINALROUTE,":",AMOUNTUOM,":",ORIGINALRATEUOM,":",LABEL,":"))
tl = tl %>% bind_rows(
  in_cohort %>% select(SUBJECT_ID, CHARTTIME, LONGLABEL, VALUE=AMOUNT) %>% mutate(VALUE = VALUE %>% as.character()) %>% rn()
) 
rm(outevents, out_cohort); gc()



#OUTPUT events
outevents = read_csv(paste0(mimicdir,"OUTPUTEVENTS.csv"), guess_max = 1e5) %>% tbl_df()
dce = read_csv(paste0(mimicdir,"D_ITEMS.csv")) %>% tbl_df()
out_cohort = outevents %>% event_filter(cohort) %>% 
  left_join(dce %>% select(ITEMID, LABEL), by="ITEMID") %>%
  mutate(LONGLABEL=paste0(LABEL,":", VALUEUOM))
tl = tl %>% bind_rows(
  out_cohort %>% select(SUBJECT_ID, CHARTTIME, LONGLABEL, VALUE) %>% mutate(VALUE = VALUE %>% as.character()) %>% rn()
) 
rm(outevents, out_cohort); gc()


# Procedure events
procevents = read_csv(paste0(mimicdir,"PROCEDUREEVENTS_MV.csv"), guess_max = 1e5) %>% tbl_df() #unfortunately the demoninator here is determined by presence in MetaVision or CareVue
proc2events = read_csv(paste0(mimicdir,"PROCEDURES_ICD.csv"), guess_max = 1e5) %>% tbl_df()
dce = read_csv(paste0(mimicdir,"D_ITEMS.csv")) %>% tbl_df()
dicd = read_csv(paste0(mimicdir,"D_ICD_PROCEDURES.csv")) %>% tbl_df()
proc_cohort = procevents %>% event_filter(cohort) %>% 
  left_join(dce %>% select(ITEMID, LABEL), by="ITEMID") %>%
  mutate(LONGLABEL=paste0("MV:",LABEL))
proc2_cohort = proc2events %>% event_filter(cohort) %>% 
  left_join(dicd %>% select(ICD9_CODE, LONG_TITLE), by="ICD9_CODE") %>%
  mutate(LONGLABEL=paste0("PROC_ICD:",LONG_TITLE)) %>%
  inner_join(adm %>% select(HADM_ID, DISCHTIME), by="HADM_ID")
tl = tl %>% bind_rows(
  proc_cohort %>% select(SUBJECT_ID, STARTTIME, LONGLABEL, STATUSDESCRIPTION) %>% mutate(LONGLABEL=paste0(LONGLABEL,":","START")) %>% rn(),
  proc_cohort %>% select(SUBJECT_ID, ENDTIME, LONGLABEL, STATUSDESCRIPTION) %>% mutate(LONGLABEL=paste0(LONGLABEL,":","END")) %>% rn()
)
tl = tl %>% bind_rows(
  proc2_cohort %>% select(SUBJECT_ID, DISCHTIME, LONGLABEL) %>% rn()
)
rm(procevents, proc2events, proc_cohort, proc2_cohort); gc()


# load transfer data
trs = read_csv(paste0(mimicdir,"TRANSFERS.csv")) %>% tbl_df()
tl = tl %>% bind_rows(
  trs %>% event_filter(cohort) %>% 
    select(SUBJECT_ID, INTIME, PREV_CAREUNIT) %>% mutate(PREV_CAREUNIT=paste0("FROM:", PREV_CAREUNIT),
                                                         VALUE = NA) %>% rn(),
  trs %>% event_filter(cohort) %>% 
    select(SUBJECT_ID, INTIME, CURR_CAREUNIT) %>% mutate(CURR_CAREUNIT=paste0("TO:", CURR_CAREUNIT),
                                                         VALUE = NA) %>% rn(),
)

svs = read_csv(paste0(mimicdir,"SERVICES.csv")) %>% tbl_df()
tl = tl %>% bind_rows(
  svs %>% event_filter(cohort) %>% 
    select(SUBJECT_ID, TRANSFERTIME, PREV_SERVICE) %>% mutate(PREV_SERVICE=paste0("FROM_SERVICE:", PREV_SERVICE),
                                                              VALUE = NA) %>% rn(),
  svs %>% event_filter(cohort) %>% 
    select(SUBJECT_ID, TRANSFERTIME, CURR_SERVICE) %>% mutate(CURR_SERVICE=paste0("TO_SERVICE:", CURR_SERVICE),
                                                              VALUE = NA) %>% rn(),
)

# make events all lower case
tl = tl %>% mutate(event = str_to_lower(event))

# make times as datetimes
tl = tl %>% mutate(t = lubridate::as_datetime(t))

# sort
tl = tl %>% arrange(pt, t, event)

# Save to file
write_csv(tl, paste0(prefix,nameExtension,"timeline.csv.gz"))

