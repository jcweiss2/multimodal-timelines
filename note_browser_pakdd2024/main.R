### Load/install libraries

loadlibs = function(libs) {
  for(lib in libs) {
    class(lib)
    if(!do.call(require,as.list(lib))) {install.packages(lib)}
    do.call(require,as.list(lib))
  }
}
libs = c("tidyverse","ggplot2","bsplus", "ggthemes", "DT", "glmnet", "shiny","data.table","zip","shinythemes",
         "shinymanager","shinyFiles","keys","shinyWidgets","gridExtra","sparsesvd","furrr")
loadlibs(libs)

library(bsplus)
library(ggplot2)
library(ggthemes)
library(DT)
library(tidyverse)
library(shiny)
library(data.table)
library(zip)
library(shinythemes)
library(shinymanager)
library(shinyFiles)

sourcedir = "<your-source-path-to-this-main-file>"
dat_file = "<your-structured-data-file-path-here>"
note_file = "<your-path>/matched.csv"
annotation_directory = "<your-annotation-save-location>"

# Demo data: 'dat' contains the structured data, replace with your tibble from 'dat_file'
dat = tibble(hosp_id=c(1,1),
             TIME=c("1970-12-11 00:00:00","1971-01-01 00:00:00") %>% lubridate::ymd_hms(),
             event=c("first","last"),
             value=c(NA,NA),
             xid="0.xml")
# delim = "|"  # alternatively, load from file
# cat(file=stderr(), getwd(), dat_file, "\n") 
# dat = fread(dat_file) %>% as_tibble() %>%
#   select(hosp_id=1, TIME=t, event=event, value=value, xid=xid)
dat = dat %>% left_join(tibble(event=dat$event %>% unique()) %>% arrange(desc(event)) %>% mutate(eventi=1:nrow(.)), by="event")

# Demo data: 'notat' contains the text data
# Example format for 'matched.csv', which contains the note text
notat = tibble(
  hadm_id=1,
  text="A 19-year-old man was admitted to the pediatric ICU because of shock, multiple organ failure, and rash. Twenty hours before admission, abdominal pain and nausea developed after he ate leftovers from a restaurant meal. Five hours before admission, purplish discoloration of the skin developed. Management decisions were made. (Excerpt from abstract: https://www.nejm.org/doi/full/10.1056/NEJMcpc2027093)
",
  xml="0.xml",
  new_admit_time="1971-01-01",
  old_admit_time="1971-01-01")
# notat = fread(note_file, strip.white=F) %>% as_tibble()

# Your annotation file (created through the viz tool) will load up your saved annotations 
# seed_annotation_file = "<your-saved-annotation-file>"
seed_annotation_file = NULL # Use this if no preloaded annotation exists

annotation_directory = "/Users/weissjc/note_browser_pakdd2024/logging"
if(!dir.exists(annotation_directory)) {
  dir.create(annotation_directory, recursive = T)
}
annotation_subdirectory = paste(annotation_directory, format(Sys.time(), "%Y-%m-%d_%H-%M-%S"),sep="/")
dir.create(annotation_subdirectory)

# Helper files
source(paste0(sourcedir,"csv_loader.R"))
source(paste0(sourcedir,"list_maker.R"))
source(paste0(sourcedir,"header.R"))
source(paste0(sourcedir,"cohort_extracting.R"))

ui = function(request) {
  tagList(
    tags$style(
      HTML("#cohort-note_contents { overflow-wrap: break-word; width=100%; padding:10px; display:block;
                           background-color:#f5f5f5; margin: 0 0 10.5px;}")
    ),
    tags$head(tags$style(HTML(get_header()))),
    navbarPage("TL-Lite", 
               tabPanel("Cohort selection", 
                        cohortModuleUI("cohort",dat)
               ),
               tabPanel("Settings",
                        shinythemes::themeSelector()
               ),
               id = "tabs",
               selected="Cohort selection",
               theme=shinythemes::shinytheme("yeti")
    )
  )
}

# Wrap your UI with secure_app
# ui = secure_app(ui)

server <- function(input, output, session) {
  
  ### Central data tibbles
  all_data = reactiveVal(value=dat)
  all_datasets = reactiveVal(data.frame(name="Demo",
                                        location="--",
                                        type="default",
                                        stringsAsFactors=F) %>% 
                               as_tibble() %>% 
                               mutate(data=list(tibble(x=1) %>% .[0,0]),
                                      columns=list(1:4),
                                      extra_data_folder_write="",
                                      extra_data_folder_read=""))

  ### Modules for the panels
  cmValues = cohortModuleServer("cohort", cohort_selection(all_data=all_data, 
                                                           all_datasets=all_datasets, 
                                                           auth=NULL,
                                                           ann_dir=annotation_subdirectory,
                                                           seed_annotation_file=seed_annotation_file))

  use_bs_tooltip()  
}

shinyApp(ui=ui, server=server,
         enableBookmarking = "server"
)

