require(shiny)
require(data.table)
require(tidyverse)
require(shinyWidgets)
require(shinyFiles)
require(bsplus)

tryNumeric = function(.) {  # Convert to numeric if does not result in NA
  ifelse(!is.na(as.numeric(.)),as.numeric(.),.)
}

myDownloadButton <- function(outputId, label = "Download"){
  tags$a(id = outputId, class = "btn btn-default shiny-download-link", href = "", 
         target = "_blank", download = NA, NULL, label)
}

csvFileUI <- function(id, label = "D: Datasets", placeholder="Data File") {
  # `NS(id)` returns a namespace function, which was save as `ns` and will
  # invoke later.
  ns <- NS(id)
  bs_delay = list(show = 1000, hide = 50)
  
  div(class = "option-group",
      div(class = "option-header", label),
      fluidRow(
        column(width=6,
          DT::dataTableOutput(ns("chosen"))
        ),
        column(width=6,
               fluidRow(
                 column(width=8,
                        textInput(ns("name"), label="Data name:", placeholder=placeholder) %>%
                          bs_embed_tooltip(title = "Give the dataset a name", delay=bs_delay)
                 ),
                 column(width=4,
                        textInput(ns("delim"), label="Delimiter", value = ",") %>%
                          bs_embed_tooltip(title = "Set the file delimiter", delay=bs_delay)
                 )
               ),
               fluidRow(
                   column(width=6,
                          textInput(ns("columns"),label="Columns", value = "1,2,3,4") %>%
                            bs_embed_tooltip(title = "Identify the 'id', 'time', 'event', and 'value' columns, by name or index", delay=bs_delay)
                   ),
                   column(width=6,
                          tagList(
                            headerPanel(" "),
                            tags$p(" "),
                            switchInput(
                              inputId = ns("heading"),
                              label = "Header",
                              value=T,
                              size="mini",
                              inline=T
                            )
                          )
                   )
                 ),
               fileInput(ns("file"), label="Upload csv or rds file (NO private data)", placeholder="4-column-file.csv", multiple=T),
               fluidRow(
                 column(width=4,
                        actionButton(ns("remove"),label="Remove")
                 ),
                 column(width=8,
                        myDownloadButton(ns("download_csv"), label="Download"),
                        switchInput(ns("download_type"),
                                    onLabel = "csv",
                                    offLabel = "rds",
                                    value=T,
                                    size="mini",
                                    inline=T) %>%
                          bs_embed_tooltip(title = "Select the download type", delay=bs_delay)
                 )
               ),
               textOutput(ns("upload_comments")),
               conditionalPanel("output.has_extra_folder_read",
                                ns=ns,
                                switchInput(ns("extra_merge"),
                                            label="Merge extra",
                                            value=F,size = "mini"
                                )
               )
        )
      )
  )
}
csvFileServer <- function(id, datasets, active_data, original_data, loading_type="csv",
                          mimic_csv_loader=NULL,auth=NULL) {
  moduleServer(
    id,
    function(input, output, session) {
      userFile <- reactive({
        validate(need(input$file, message = F))
        input$file
      })
      
      reselection = reactive({
        if(!is.null(mimic_csv_loader)) {
          return(mimic_csv_loader$rows_selected())
        }
        return(previously_selected$selection)
      })
      previously_selected=reactiveValues(selection=NULL)
      observe({
        ds = datasets()
        previously_selected$selection=as.numeric(isolate(selected()))
      })
      
      output$chosen <- DT::renderDataTable({
        datatable(datasets() %>% select(-location, -data, -extra_data_folder_write, -extra_data_folder_read), 
                  options=list(dom="t", compact=T),
                  autoHideNavigation=T,
                  selection=list(mode="single",target="row",selected=reselection()),
                  rownames = F)
      })
      chosen_proxy = DT::dataTableProxy("chosen")
      
      onStart = reactiveVal(value=T)
      observe({
        if(onStart()) {
          chosen_proxy %>% selectRows(1)
          onStart(F)
          if(!is.null(auth)) {
            ucomments(paste("User", auth$user, "max file size is", 
                            options()$shiny.maxRequestSize/1024/1024, "Mb"))
          } else {
            ucomments(paste("Max file size is", options()$shiny.maxRequestSize/1024/1024, "Mb"))
          }
        }
      })
      
      observe({  # Update csv_loader to agree with previous panel  # will be overwritten if those change but not vice versa
        if(!is.null(mimic_csv_loader)) {
          updateTextInput(session=session,"delim",value=mimic_csv_loader$delim())
          updateSwitchInput(session=session,"heading",value=mimic_csv_loader$heading())
        }
      })
      
      # The user's data, parsed into a data frame
      datatype = reactive({
        if(str_detect(userFile()$name, pattern="\\.rds(\\.gz)?$")) {
          print("rds")
          "rds" 
        } else if (str_detect(userFile()$name, pattern="sv(\\.gz)?$")) {        
          print("csv")
          "csv"
        } else {
          validate("Unrecognized file type")
        }
      })
      
      dataframe <- reactive({
        validate(need(!is.null(datatype()), F))
        validate(need(datatype()=="csv", F))
        fread(userFile()$datapath, header=input$heading, sep=isolate(input$delim),
              integer64="double") %>% as_tibble()
      })
      
      output$upload_comments = reactive({
        ucomments()
      })
      ucomments = reactiveVal(value="")
      observe({
        df = dataframe()
        if(is.null(df)) {
          ucomments("")
        } else if(ncol(df)<4) {
          ucomments("Failed to parse 4 columns")
        } else {
          ucomments("")
        }
      })

      observe({
        datapath = userFile()$datapath
        validate(need(str_split(input$columns,",") %>% .[[1]] %>% length() >= 4, "Please provide >=4 comma separated columns")
        )
        df = dataframe()
        validate(need(ncol(df)>=4, "Failed to parse >=4 columns"))
        isolate({
          datasets(datasets() %>% bind_rows(data.frame(name=isolate(input$name),
                                                       location=userFile()$datapath,
                                                       type="csv",
                                                       stringsAsFactors = F) %>%
                                              as_tibble() %>% 
                                              mutate(data=list(df),
                                                     columns=lapply(str_split(input$columns,","), str_trim),
                                                     extra_data_folder_write = NA,
                                                     extra_data_folder_read = NA)
          ))
          if(nrow(datasets())==1) {
            chosen_proxy %>% selectRows(1)
          }
          msg = sprintf("File %s was uploaded", userFile()$name)
          cat(msg, "\n")
        })
      })
      
      observe({
        datapath = userFile()$datapath
        validate(need(!is.null(datatype()),F))
        if(datatype()=="rds") {
          isolate({
            rdsobj = read_rds(datapath)
            datasets(datasets() %>% bind_rows(rdsobj))
          })
        }
      })
      
      output$has_extra_folder_read = reactive({
        has_extra_folder_r()
      })
      outputOptions(output,"has_extra_folder_read", suspendWhenHidden=F)
      has_extra_folder_r = reactiveVal(value=F)
      
      observe({
        ds = datasets()
        rows_selected = selected()
        isolate({
          validate(need(!is.null(ds),F))
          validate(need(nrow(ds)>0,F))
          validate(need(!is.null(rows_selected),F))
          if(is.na(ds[[selected(),"extra_data_folder_read"]]) ||
             ds[[selected(),"extra_data_folder_read"]]=="") {
            has_extra_folder_r(F)
          } else {
            has_extra_folder_r(T)
          }
        })
      })
      
      expand_event_with_eventi = function(event_df) {
        event_eventi = event_df %>% select(event) %>% distinct() %>% 
          arrange(desc(event)) 
        if(nrow(event_eventi)>0) { # 0 rows error avoidance
          event_eventi = event_eventi %>% mutate(eventi=1:nrow(.)) 
        } else {
          event_eventi = event_eventi %>%
            bind_rows(data.frame(event="",stringsAsFactors=F)) %>% mutate(eventi=1) %>% .[0,]
        }
        return(event_df %>% left_join(event_eventi, by="event"))
      }
      
      # Change data set
      observeEvent(selected(),{
        rows_selected = selected()
        rows_selected = ifelse(is.null(rows_selected),1,rows_selected)
        isolate({
            active_data(original_data())
        })
      })
      
      observeEvent(input$remove, {
        validate(need(!is.null(input), message="Please choose a row to remove"))  # need to separate csv and chunkfile modules
        row_selected = input$chosen_rows_selected
        if(nrow(datasets())<=row_selected) {
          chosen_proxy %>% selectRows(NULL)
        }
        datasets(datasets()[-row_selected,])
      })
      
      selected = reactive({
        rows_selected = input$chosen_rows_selected
        if(is.null(rows_selected)) { return(NULL) }
        result = suppressWarnings(min(nrow(datasets()), rows_selected))
        if(is.infinite(result)) { return(NULL) } else { return(result) }
      })
      
      # Download a row as an RDS i.e. saveRDS
      output$download_csv = downloadHandler(
        filename = function() {
          paste0(datasets()[[selected(),"name"]],"_extract.",
                 ifelse(input$download_type, "csv","rds"),".gz")
        },
        content = function(file) {
          if(input$download_type) {  #csv
            if(datasets()[[selected(),"type"]]=="csv") {
              write_csv(datasets()[[selected(),"data"]][[1]], file)
            } else if(datasets()[[selected(),"type"]]=="default") {
              write_csv(data.frame(x=1:3) %>% as_tibble() %>% .[0,], file)
            } else if(datasets()[[selected(),"type"]] %in% c("version","csv_version")) {
              if(input$extra_merge) {
                data1 = datasets()[[selected(),"data"]][[1]]$data_handler
                browser()
                extra_folder = datasets()[[selected(), "extra_data_folder_read"]]
                delim = input$delim
                header = input$heading
                original_suffix = str_replace(datasets()[[selected(), "location"]], ".*\\.","") %>% paste0("$")
                original_suffix = str_replace(datasets()[[selected(), "location"]], ".*\\.","") %>% 
                  str_replace(".gz$","") %>% paste0(".gz$")
                require(furrr)
                data2 = data.frame(location=list.files(extra_folder)) %>%
                  as_tibble() %>% filter(str_detect(location, pattern=original_suffix)) %>%
                  mutate(data = future_map(location, ~
                                             fread(paste0(extra_folder,"/",.x), 
                                                   sep=delim,
                                                   header=header,
                                                   integer64="double") %>%
                                             as_tibble() %>%
					     mutate(TIME=parse_guess(as.character(TIME)))
					     ))
                data = bind_rows(data1 %>%
                                   mutate(across(where(is.numeric) & (value | value_original), as.character)),
                                 data2 %>% select(data) %>% unnest(everything()) %>%
                                   mutate(across(where(is.numeric) & (value | value_original), as.character)))
              } else {
                data = datasets()[[selected(),"data"]][[1]]$data_handler
              }
              write_csv(data, file)
            }
          } else {  # rds
            saveRDS(datasets()[selected(),], file)
          }
        }
      )
      
      # Return the reactive that yields the data frame
      return(list(dataframe=dataframe,
                  rows_selected=selected,
                  delim=reactive(input$delim),
                  heading=reactive(input$heading),
                  merge_flag = reactive(input$extra_merge)))
    }
  )
}
