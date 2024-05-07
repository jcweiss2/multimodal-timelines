library(stats)
library(ggthemes)
library(DT)
library(tidyverse)
library(zip)
require(Matrix)
require(sparsesvd)
require(gridExtra)
require(bsplus)
require(keys)

cohortModuleUI <- function(id, start_data) {
  # `NS(id)` returns a namespace function, which was save as `ns` and will
  # invoke later.
  ns <- NS(id)
  bs_delay = list(show = 1000, hide = 50)
  bs_delay_2 = list(show = 2000, hide = 50)
  bs_delay_4 = list(show = 4000, hide = 50)
  bs_delay_slow = bs_delay_4
  
  js = '      function getSelectionText() {
                var text = window.getSelection().toString();
                var range = window.getSelection().getRangeAt(0);
                var start_offset = range.startOffset;
                var end_offset = range.endOffset;
                return {"text":text, "start":start_offset, "end":end_offset};
              }
              showSelection = function() {
                  var selection = getSelectionText();
                  Shiny.onInputChange("#id#-selected_text", selection);
              }
              
              document.getElementById("#id#-note_contents").onmouseup = showSelection
              
              ' %>%
    str_replace_all("#id#", id)

  fluidPage(
    
    useKeys(),
    
    fluidRow(column(width=6,
                    csvFileUI(ns("datafile"), "D: Datasets")),
             column(width=6,
                    div(class = "option-group",
                        div(class="option-header", "X: Explore"),
                        fluidRow(
                          column(width=4,
                                 tagList(
                                   selectInput(ns("pts"), "Patient", NULL,
                                                  selected = NULL),
                                   radioButtons(ns("zoom"), "Zoom on selection",choiceNames=c("Zoom","Hold","Reset"), 
                                                choiceValues=c("zoom","hold","reset"), selected = "hold") %>%
                                     bs_embed_tooltip(title = "Zoom action to apply for brushing events", delay=bs_delay_slow),
                                   fluidRow(
                                     column(width=6,
                                       sliderInput(ns("lb_zoom_out_exponent"), label="Zoom: (L)", min = 0, max=5, value = 0, ticks=F, width="100px"),
                                     ),
                                     column(width=6,
                                            sliderInput(ns("ub_zoom_out_exponent"), label="(R)", min = 0, max=5, value = 0, ticks=F, width="100px"),
                                     )
                                   ),
                                   fluidRow(
                                     column(width=6,
                                            actionButton(ns("undo_zoom"), "Back") %>%
                                              bs_embed_tooltip(title = "Go back a view", delay=bs_delay)
                                     ),
                                     column(width=6,
                                            actionButton(ns("enter_debugger"),"Debug") %>%
                                              bs_embed_tooltip(title = "Enter debugger (local runtime only)", delay=bs_delay_2),       
                                     )
                                   ),
                                   
                                   # bookmarkButton(),
                                   div(class = "option-group",
                                       div(class = "option-header", "T: Windowing"),
                                       fluidRow(
                                         column(width=6,
                                                textInput(ns("startat"),"Start at Event") %>%
                                                  bs_embed_tooltip(title = "The first of this event is the lower window bound", delay=bs_delay_2)
                                         ),
                                         column(width=6,
                                                textInput(ns("uptofirst"),"Up To Event") %>%
                                                  bs_embed_tooltip(title = "The first of this event is the upper window bound", delay=bs_delay_2)
                                         )
                                       ),
                                       fluidRow(
                                         column(width=6,
                                                switchInput(ns("include_prelb"), 
                                                            onLabel="Pre-", 
                                                            offLabel="Off",
                                                            value=T, 
                                                            labelWidth="100%", size="mini") %>%
                                                  bs_embed_tooltip(title = "Retain features prior to lower window bound when versioned", delay=bs_delay)
                                         )
                                       )
                                   )
                                 )
                          ),
                          column(width=8,
                                 div(id="brushopts", style="padding: 10px",
                                     div(class = "option-header", "Brush"),
                                     radioButtons(ns("brush_dir"), "Brush direction(s)",
                                                  c("x", "y", "xy"), selected="x", inline = TRUE) %>%
                                       bs_embed_tooltip(title = "Select an axis for brushing points", delay=bs_delay_2),
                                     div(class = "option-header", "Value/annotation plots"),
                                     fluidRow(
                                       column(width=6,
                                              checkboxInput(ns("varinspect_toggle"), "View values") %>%
                                                bs_embed_tooltip(title = "View values of selected events", delay=bs_delay),
                                              checkboxInput(ns("varannotate_toggle"), "View annotations") %>%
                                                bs_embed_tooltip(title = "View values of selected events", delay=bs_delay),
                                              checkboxInput(ns("varinspect_drag_add"), "Drag to view/add") %>%
                                                bs_embed_tooltip(title = "Enable event selection by brushing points", delay=bs_delay)
                                       ),
                                       column(width=6,
                                              conditionalPanel("input.event_color=='Group'", ns=ns,
                                                               sliderInput(ns("ncenters"), "Groups", min = 2, max=100, value=7)
                                              )
                                       )
                                     ),
                                     conditionalPanel("input.varinspect_toggle", ns=ns,
                                                      pickerInput(
                                                        inputId = ns("varinspect_event"),
                                                        label = "Variable", 
                                                        choices="",
                                                        multiple=T, 
                                                        options = list(
                                                          `live-search` = T,
                                                          `actions-box` = T,
                                                          size=100,
                                                          virtualScroll=100)
                                                      ) %>%
                                                        bs_embed_tooltip(title = "Select events to display", delay=bs_delay_2),
                                     )
                                 ),
                                 fluidRow(
                                   column(width=6,
                                          radioGroupButtons(
                                            inputId = ns("event_order"),
                                            label = "Event ordering",
                                            choices = c("Alpha", 
                                                        "Auto"),
                                            selected = "Alpha"
                                          ) %>% 
                                            bs_embed_tooltip(title = "Choose the event ordering (y-axis)", delay=bs_delay_2)
                                   ),
                                   column(width=6,
                                          radioGroupButtons(
                                            inputId = ns("event_color"),
                                            label = "Event coloring",
                                            choices = c("X/Y", 
                                                        "Group"),
                                            selected = "X/Y"
                                          ) %>%
                                            bs_embed_tooltip(title = "Choose the event coloring", delay=bs_delay_2)
                                   )
                                 ),
                                 fluidRow(
                                   column(width=12,
                                          checkboxInput(ns("annotation_hide_nonanchors"),"Hide structured data", value=F)
                                   )
                                 ),
                                 fluidRow(
                                   column(width=6,
                                          radioGroupButtons(
                                            inputId = ns("annotation_bound_mode"),
                                            label = "Annotation mode",
                                            choices = c("Bounds", 
                                                        "Probability"),
                                            selected = "Bounds"
                                          ) %>% 
                                            bs_embed_tooltip(title = "Annotate bounds or probabilites", delay=bs_delay_2)
                                   ),
                                   column(width=3,
                                          checkboxInput(ns("annotation_lb_use"),"Use LB", value=T)
                                   ), 
                                   column(width=3,
                                          checkboxInput(ns("annotation_ub_use"),"Use UB", value=T)
                                   )
                                 ),
                                 conditionalPanel("input.annotation_bound_mode=='Probability'", ns=ns,
                                                  fluidRow(
                                                    column(width=6,
                                                           sliderTextInput(ns("slider_lb_uncertainty"),
                                                                           "Lower uncertainty sd",
                                                                           choices = c("1s", "3s", "10s", "1M", "3M", "10M", "30M","1h", "2h", "4h", "6h", "12h", "24h",
                                                                                       "2d", "1w", "2w", "1m", "1y", "10y", "100y"),
                                                                           selected="1h")
                                                    ),
                                                    column(width=6,
                                                           sliderTextInput(ns("slider_ub_uncertainty"),
                                                                           "Upper uncertainty sd",
                                                                           choices = c("1s", "3s", "10s", "1M", "3M", "10M", "30M","1h", "2h", "4h", "6h", "12h", "24h",
                                                                                       "2d", "1w", "2w", "1m", "1y", "10y", "100y"),
                                                                           selected="1h"),
                                                           
                                                    ),
                                                    
                                                  ),
                                                  fluidRow(
                                                    column(width=6,
                                                           sliderTextInput(ns("slider_duration_uncertainty"),
                                                                           "Duration uncertainty sd",
                                                                           choices = c("1s", "3s", "10s", "1M", "3M", "10M", "30M","1h", "2h", "4h", "6h", "12h", "24h",
                                                                                       "2d", "1w", "2w", "1m", "1y", "10y", "100y"),
                                                                           selected="1h"),
                                                           plotOutput(ns("duration_distribution"),height=150, width = "100%")
                                                    ),
                                                    column(width=6,
                                                           radioGroupButtons(
                                                             inputId = ns("annotation_lbub_tied"),
                                                             label = "Uncertainty ties",
                                                             choices = c("On", 
                                                                         "Off"),
                                                             selected = "On"
                                                           ) %>% 
                                                             bs_embed_tooltip(title = "Tie the bound uncertainties", delay=bs_delay_2)
                                                    )
                                                  )
                                 )
                          ))
                    )
             )
    ),
    fluidRow(
      uiOutput(ns("plotui")),
      conditionalPanel("input.varinspect_toggle", ns=ns,
                       uiOutput(ns("varinspectui"))
      ),
      conditionalPanel("input.varannotate_toggle", ns=ns,
                       uiOutput(ns("varannotateui"))
      )
    ),
    fluidRow(
      column(width=6,
             textOutput(ns("note_contents"))
      ),
      column(width=6,
             h4("Selected text:"),
             verbatimTextOutput(ns("note_highlighted")),
             div(class = "option-group",
                 fluidRow(
                   column(width=3,
                          actionButton(label="Annotate",ns("record_annotation")),
                   ),
                   column(width=3,
                          checkboxInput(ns("annotation_selection_negation"),label="Negated",value=F)
                   ),
                   column(width=6,
                          textInput(ns("save_annotator"),label = NULL, placeholder = "Annotator name"),
                   ),
                 )
             ),
             wellPanel(
               h4("Events selected in region:"),
               DT::dataTableOutput(ns("plot_brushed_points"))
             ),
             wellPanel(
               DT::dataTableOutput(ns("selected_note_annotations")),
               DT::dataTableOutput(ns("note_annotations")),
               actionButton(ns("note_annotations_remove"), label="Remove selected rows"),
               actionButton(ns("note_annotations_remove_all"), label="all"),
               actionButton(ns("note_annotations_clear_selection"), label="Deselect all")
             ),
             div(class = "option-group",
                 fluidRow(
                   column(width=4,
                          actionButton(label="Save annotations",ns("save_annotations"))
                   ),
                   column(width=8,
                          textInput(ns("save_annotation_suffix"),label=NULL, placeholder = "Annotation file suffix"),
                   )
                 ),
                 fluidRow(
                   verbatimTextOutput(ns("save_annotation_response"))
                 )
             )
      )
    ),
    tags$script(js)
  )
}  

cohortModuleServer <- function(id, f) {
  moduleServer(
    id,
    f
  )
}

cohort_selection = function(all_data,
                            all_datasets,
                            auth=NULL,
                            ann_dir=NULL,
                            seed_annotation_file=NULL
                            ) {
  return(function(input, output, session) {
    # cat(file=stderr(), "here")
    all_ids = reactiveVal(value=NULL)
    # cat(file=stderr(), "HERE")
    all_events = reactiveVal(value=NULL)
    all_outcomes = reactiveVal(value=NULL)
    all_inclusions = reactiveVal(value=NULL)
    all_exclusions = reactiveVal(value=NULL)
    all_features = reactiveVal(value=NULL)
    all_annotations = reactiveVal(value=NULL)
    all_events_and_annotations = reactiveVal(value=NULL)

    all_note_annotations = reactiveVal(value=NULL)
    
    all_data = all_data
    original_data_stored = reactiveVal(F)
    original_data = reactiveVal(NULL)
    observe({
      validate(need(!original_data_stored(),"ok to store"))
      original_data_stored(T)
      original_data(all_data())
    })

    selected_events = reactive({  
      validate(need(input$varinspect_drag_add, F))      
      zp = zoom_pts()
      pb = input$plot_brush
      
      validate(need(!is.null(zp),F))
      
      if(yvar() == "auto") {
        zp[[1]] %>% left_join(auto_map(), by="eventi") %>% 
          mutate(Assignment = forcats::fct_explicit_na(Assignment, na_level="Group NA")) %>%
          mutate(auto=ifelse(is.na(combined),-1,combined)) %>%
          brushedPoints(input$plot_brush) %>% as_tibble() %>% select(event) %>% distinct()
      } else {
        zp[[1]] %>% brushedPoints(pb) %>% as_tibble() %>% select(event) %>% distinct()
      }
    })
    
    
    output$selected_note_annotations <- DT::renderDataTable({
      anns = all_note_annotations()
      local_curpt = curpt()
      validate(need(!is.null(anns),F))
      validate(need(!is.null(local_curpt),F))
      
      dat = all_note_annotations() %>% 
        filter(pt == as.character(local_curpt$hosp_id[1])) %>% 
        select(text, bounds, negated, text_position) %>% 
        mutate(bounds = modify_if(bounds, ~ !is_tibble(.x), ~ .x[[1]])) %>% 
        unnest(everything()) 
      
      if(nrow(dat)==0) { return(NULL) }
      dat = dat %>%
        select(text, negated, interval.lb=lb, interval.ub=ub, start, end) %>% 
        mutate(note.text = filter(notat, hadm_id==input$pts)[["text"]]) %>%
        mutate(context = substr(note.text,
                                start-30,
                                end+30)
        ) %>%
        select(-note.text) %>%
        mutate(interval.lb = lubridate::as_datetime(interval.lb),
               interval.ub = lubridate::as_datetime(interval.ub)) %>% 
        mutate(dur=interval.ub-interval.lb) %>%
        mutate(is.interval=T) %>%
        mutate(duration = lubridate::time_length(dur, "hours"))
      
      brushedPoints(dat, input$annotate_plot_brush)
    })
    
    # Modules
    csvFileLoader = csvFileServer("datafile", all_datasets, all_data, original_data, auth=auth)
   
    observeEvent(input$enter_debugger, {
      browser()
    })
    
    previous_csv_rows_selected = reactiveVal(value=-1)
    
    observe({  
      rows_selected = csvFileLoader$rows_selected()
      validate(need(!is.null(rows_selected),F))  # don't update if no row is selected
      validate(need(rows_selected!=previous_csv_rows_selected(),F))  # don't update if no row is selected
      previous_csv_rows_selected(rows_selected)
      isolate({
        data = all_data()
        validate(need(!is.null(input$pts), "Cannot update selection: no patient selected"),
                 need(!is.null(data), "No data found"))
      
        if(all_datasets()[[rows_selected,"type"]] %in% c("version","csv_version")) {
          vo = all_datasets()[[rows_selected,"data"]][[1]]
          all_ids(all_data()[,"hosp_id"] %>% distinct())
          all_events(all_data()[,"event"] %>% distinct())
          all_features(vo$features)
          all_outcomes(vo$outcomes)
          all_inclusions(vo$inclusions)
          all_exclusions(vo$exclusions)
          all_annotations(vo$annotations)
        } else if(all_datasets()[[rows_selected,"type"]] == "csv") {
          all_ids(all_data()[,"hosp_id"] %>% distinct())
          all_events(all_data()[,"event"] %>% distinct())
          all_features(all_data()[,"event"] %>% .[0,])
          all_outcomes(all_data()[,"event"] %>% .[0,])
          all_inclusions(all_data()[,"event"] %>% .[0,])
          all_exclusions(all_data()[,"event"] %>% .[0,])
          all_annotations(all_data() %>% head(1) %>% select(event) %>%
                            mutate(name="", condition="", marked=F, lagtime=0) %>% 
                            .[0,c(2,1,3,4,5)])
        } else {
          all_ids(all_data()[,"hosp_id"] %>% distinct())
          all_events(all_data()[,"event"] %>% distinct())
          all_features(all_data()[,"event"] %>% .[0,])
          all_outcomes(all_data()[,"event"] %>% .[0,])
          all_inclusions(all_data()[,"event"] %>% .[0,])
          all_exclusions(all_data()[,"event"] %>% .[0,])
          all_annotations(all_data() %>% head(1) %>% select(event) %>%
                            mutate(name="", condition="", marked=F, lagtime=0) %>% 
                            .[0,c(2,1,3,4,5)])
        }
        
        
        updateSelectInput(session, 
                          "pts",
                          choices=all_ids()$hosp_id,
                          selected = ifelse(input$pts %in% all_ids()$hosp_id,
                                            input$pts,
                                            all_ids()$hosp_id[1])
                          # server=T
                          )
        zoom_pts(rlang::list2(data %>% filter(hosp_id==input$pts) %>% 
                                filter(TIME!=min(TIME, na.rm=T), TIME!=max(TIME, na.rm=T))))
        
      })
    })
    
    observe({
      events = all_events()
      annotations = all_annotations()
      validate(need(!is.null(events),F),
               need(!is.null(annotations),F))
      all_events_and_annotations(events %>% bind_rows(annotations %>% select(event=name)))
    })
    
    observe({
      rows_selected = input$datafile$rows_selected
      rows_selected = ifelse(is.null(rows_selected),1,rows_selected)
      isolate({
        all_data(original_data())
      })
    })

    observeEvent(input$pts, {
      
    })
    
    observeEvent(input$annotate_add,{
      annotations_name = isolate(input$annotate_name)
      annotations_event = isolate(input$annotate_make)
      annotations_condition = isolate(input$annotate_condition)
      annotations_point = isolate(input$annotate_truth)
      annotations_lag = isolate(input$annotate_lagtime)
      events = isolate(all_events())
      if(annotations_name != "") {
        all_annotations(all_annotations() %>%
                          bind_rows(data.frame(name=annotations_name,
                                               event=annotations_event,
                                               condition=annotations_condition,
                                               marked=!annotations_point,
                                               lagtime=annotations_lag,
                                               stringsAsFactors = F) %>% as_tibble()) %>%
                          distinct()
        )
      }
    })
    observeEvent(input$annotate_remove,{
      annotations_text = isolate(input$annotate_name)
      rows_to_remove = isolate(input$annotations_chosen_rows_selected)
      events = all_annotations()
      validate(need(!is.null(events), "no annotations found"))
      if(annotations_text != "") {
        all_annotations(events %>% filter(!str_detect(name, pattern=annotations_text)))
      } else {
        if(!is.null(rows_to_remove) && length(rows_to_remove)>0) {
          all_annotations(all_annotations() %>% slice(-rows_to_remove))
        }
      }
    })

    updateProgress = function(progress, detail=NULL, value=NULL, returnProgress=F) {
      if(is.null(progress)) { return() }
      progress$set(value=value, detail=detail)
      if(returnProgress) { return(progress) }
    }


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
    
    version_object = reactiveVal(value=NULL)
    
    # ### Visualization panels ###
    zoom_pts = reactiveVal(value=NULL)

    observe({
      validate(need(!is.null(all_data()),"No data found"))
      zoom_pts(rlang::list2(isolate(all_data()) %>% 
                              filter(hosp_id==isolate(all_data())$hosp_id[1]) %>% 
                              filter(TIME>min(TIME,na.rm=T), TIME<max(TIME,na.rm=T))))
    })
    
    observe({
      addKeys(session$ns("kb_shortcuts"), c("shift+space"))
    })
    kb_record_annotation = reactiveVal(value=NULL)
    observeEvent(input$kb_shortcuts, {
      print(input$kb_shortcuts)
      if(input$kb_shortcuts=="shift+space") {
        if(is.null(kb_record_annotation)) { 
          kb_record_annotation(0)
        } else {
          kb_record_annotation(kb_record_annotation()+1)
        }
      }
    })
						
    observe({
      local_curpt = curpt()  # only show events from current patient as per request from GF
      updatePickerInput(session, "varinspect_event",  # formerly updateSelectInput
                        choices = (local_curpt %>% .[["event"]]),
                        choicesOpt = list(
                          content = 
                            local_curpt %>%
                            mutate(event = str_trunc(event, 50)) %>%
                            .[["event"]]
                        )
      )
      updatePickerInput(session, "annotate_make",
                        choices = (local_curpt %>% .[["event"]]),
                        choicesOpt = list(
                          content = 
                            local_curpt %>%
                            mutate(event = str_trunc(event, 50)) %>%
                            .[["event"]]
                        )
      )
    })


    observeEvent(input$pts, {
      validate(need(!is.null(all_data()),"No data found"))
      new_pts = isolate(all_data()) %>% filter(hosp_id==input$pts) %>% filter(TIME!=min(TIME, na.rm=T), TIME!=max(TIME, na.rm=T))
      if(nrow(new_pts)==0) {
        new_pts = isolate(all_data()) %>% filter(hosp_id==input$pts)
      }
      zoom_pts(rlang::list2(new_pts))
    })

    observeEvent(input$undo_zoom,{
      new_pts = zoom_pts()
      if(length(new_pts)==1)
        new_pts = new_pts
      else
        new_pts = zoom_pts() %>% .[2:length(.)]
      zoom_pts(new_pts)
    })
    observeEvent(c(input$zoom, input$undo_zoom,input$plot_brush), {
      if(input$zoom=="zoom") {  # 1 zoom
        if(yvar() == "auto") {
          zp = zoom_pts()[[1]] %>% left_join(auto_map(), by="eventi") %>% 
            mutate(Assignment = forcats::fct_explicit_na(Assignment, na_level="Group NA")) %>%
            mutate(auto=ifelse(is.na(combined),-1,combined))
          new_pts = zp %>% brushedPoints(input$plot_brush) %>% as_tibble()
          new_pts = new_pts %>% select(-Assignment, -auto, -combined)
        } else {
          zp = zoom_pts()[[1]]
          new_pts = zp %>% brushedPoints(input$plot_brush) %>% as_tibble()
        }
        if(nrow(new_pts) == 0)
          new_pts = zoom_pts()
        else
          new_pts = zoom_pts() %>% purrr::prepend(rlang::list2(new_pts))
        session$resetBrush(session$ns("plot_brush"))
      } else if(input$zoom=="hold") { # 2 hold
        new_pts = zoom_pts()
      } else if(input$zoom=="reset") { # 3 reset
        new_pts = all_data() %>% filter(hosp_id==input$pts) %>% rlang::list2()
        isolate({
          updateRadioButtons(session,"zoom", selected="zoom")
          session$resetBrush(session$ns("plot_brush"))
        })
      } else {
        new_pts = zoom_pts()
      }
      # zoom_pts = dat
      zoom_pts(new_pts)
    })

    # Currently selected dataset
    curpt = reactive({
      validate(need(!is.null(input$uptofirst), "need uptofirst"),
               need(!is.null(input$startat), "need startat"))
      # mtcars = mtc
      # mtcars
      # if(input$zoom=="zoom") {
      #   pts = brushedPoints(zoom_pts(), input$plot_brush)
      #   if(nrow(pts)>0) {
      #     dat = pts
      #   } else {
      #     dat = zoom_pts()
      #   }
      # } else {
      #   dat = zoom_pts()
      # }
      dat = zoom_pts()
      validate(need(!is.null(dat), F))
      dat = dat[[1]] %>% bind_rows(curannotation())
      
      if(input$annotation_hide_nonanchors) {
        dat = dat %>% filter(str_detect(event, "^admi|^discharged|^discharge_location|^chart:admi|^birthed|^gender|^ethnic"))
      }
      if(input$uptofirst != "" && (input$uptofirst %in% dat$event))  # Upper bound
        dat = dat %>% filter(TIME <= (dat %>% filter(event==input$uptofirst) %>% .[[1,"TIME"]]))
      if(input$startat != "" && (input$startat %in% dat$event) && (!input$include_prelb))  # Lower bound
        dat = dat %>% filter(TIME >= (dat %>% filter(event==input$startat) %>% .[[1,"TIME"]]))
      dat
    })
    
    curlbub = reactive({
      dat = curpt()
      validate(need(!is.null(dat), F))
      ub = ifelse(input$uptofirst %in% dat$event, dat %>% filter(event==input$uptofirst) %>% .[[1,"TIME"]], Inf)
      lb = ifelse(input$startat %in% dat$event, dat %>% filter(event==input$startat) %>% .[[1,"TIME"]], -Inf)
      return(c(lb,ub))
    })

    curannotation = reactive({
      dat = zoom_pts()[[1]]
      annotation_rules = all_annotations()
      validate(need(!is.null(dat),"no pt selected"),
               need(!is.null(annotation_rules), "no annotations found"))
      dat = dat %>% filter(event %in% annotation_rules$event) 
      
      if(nrow(dat)>0) {
        dat = dat %>%
          left_join(annotation_rules, by=c("event")) %>%
          mutate(value=suppressWarnings(
            map2_chr(value, condition, ~ glue::glue(paste0("{as.numeric(",
                                                           ifelse(.x=="",NA,
                                                                  ifelse(!is.na(as.numeric(.x)),.x,NA)),
                                                           ")",.y,"}"), .na=NA))
          )) %>%
          filter(value=="TRUE" | marked) %>%
          mutate(TIME = TIME + lagtime) %>%
          mutate(eventi = all_events() %>% nrow() + 1) %>%
          select(-event) %>%
          rename(event=name) %>%
          select(hosp_id, TIME, event, value, eventi)
      }
      return(dat)
    })

    # Name of the x, y, and faceting variables
    xvar <- reactive({ "TIME" })

    yvar <- reactive({
      if(input$event_order=="Alpha") {
        return("eventi")
      } else if(input$event_order=="Auto") {
        return("auto")
      } else {
        validate(need(F,"Invalid event_order"))
      }
      })
    
    
    ### Create groupings of events based on a sample of their time-patient cooccurrences
    auto_map = reactive({
      validate(need(yvar()=="auto", F))
      data = all_data()
      dat = isolate(curpt())
      
      rsample_n = repeatable(sample_n, 12345)
      rsample1 = repeatable(sample, 23456)
      rsample2 = repeatable(sample, 34567)
      rkmeans = repeatable(stats::kmeans, 45678)
      
      # use a limited data set for speed
      small_dat = dat %>% bind_rows(data %>% filter(hosp_id %in% (all_ids() %>% 
                                                                    rsample_n(100, replace=T))))
      
      # sample an adjacency matrix
      adj = 
        small_dat %>% select(eventi, hosp_id, TIME) %>% group_by(hosp_id, TIME) %>%
        nest() %>%
        mutate(pairs = map(data, ~ crossing(a=.x$eventi %>% rsample2(size=min(10,nrow(.x))) %>% head(10),
                                            b=.x$eventi %>% unique()))) %>%
        select(pairs) %>% unnest(everything()) %>% ungroup() %>% 
        group_by(a,b) %>% summarise(count=n()) %>% ungroup()

      # create a mapping to a temporary indexer 'spi'
      spi_map = adj %>% select(b) %>% distinct() %>% mutate(spi=1:nrow(.))
      
      adj = adj %>% 
        left_join(spi_map, by="b") %>%
        rename(bs = spi) %>%
        rename(as = a) %>%  
        select(as, bs, count)
      
      # compute svd of adjacency to get per-event signature, and cluster on it.
      require(Matrix)
      require(sparsesvd)
      sM = sparseMatrix(i=adj$as,j=adj$bs, x=adj$count*1.)
      svs = sparsesvd(sM, rank=min(max(adj$as), 10))$v
      
      assignment = rkmeans(svs, centers=min(max(adj$as), input$ncenters, nrow(unique(svs))-1), nstart=5) %>% .[["cluster"]]
      assignment_map = data.frame(spi=1:length(assignment),assignment=assignment) %>% as_tibble()
      
      clmap = small_dat %>% select(event, eventi) %>% distinct() %>%
        left_join(spi_map, by=c("eventi"="b")) %>%
        left_join(assignment_map, by=c("spi")) %>%
        select(eventi, assignment) %>%
        mutate(combined = factor(paste(assignment,":",eventi)) %>% as.numeric(),
               assignment=paste("Group",assignment) %>% factor()) %>%
        rename(Assignment=assignment)
        
      return(clmap)
    })

    output$plotui <- renderUI({
      ns = session$ns
      plotOutput(ns("plot"), height=300, width = "100%",
                 hover = hoverOpts(
                   id = ns("plot_hover"),
                 ),
                 brush = brushOpts(
                   id = ns("plot_brush"),
                   delay = input$brush_delay,
                   delayType = input$brush_policy,
                   direction = input$brush_dir,
                   resetOnNew = F  # ns("brush_reset")
                 )
      )
    })

    output$note_contents = renderText({
      cur_id = input$pts
      notat %>% filter(hadm_id==cur_id) %>% .[["text"]]
    })
    
    output$note_highlighted = renderText({
      st = input$selected_text$text
      if(!is.null(st)) {
        if(nchar(st)==0) {
          "No selection"
        } else {
          st
        }
      }
    })
    
    output$duration_distribution <- renderPlot({
      d_sd = lubridate::as.duration(input$slider_duration_uncertainty)
      if (!is.null(input$plot_brush)) {
        if("numeric" %in% class(curpt()$TIME[[1]])) {
          brush_lb = lubridate::as_datetime(input$plot_brush$xmin)
          brush_ub = lubridate::as_datetime(input$plot_brush$xmax)
          warning("Interval not a datetime so adding duration may cause NAs")
        } else {
          brush_lb = input$plot_brush$xmin
          brush_ub = input$plot_brush$xmax
        }
      } else {
        warning("No interval selected, so no annotation made")
        return(NULL)
      }
      d_mu = lubridate::as.duration(brush_ub-brush_lb)
      tpoints = lubridate::as.duration(seq(max(0, d_mu-3*d_sd), d_mu+3*d_sd, length.out=100))
      ypoints = as.numeric(
        dnorm(tpoints, mean = d_mu, sd= d_sd) / (1 - pnorm(0, d_mu, d_sd))
      )
      # browser()
      ggplot(data = tibble(density=ypoints, t=tpoints)) +
        geom_line(aes(y=density, x=t)) + ylab("") + xlab("") +
        scale_x_time() +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        ggtitle(paste0("Mean: ", lubridate::as.period(d_mu %>% round()),
                       "\n1%: ", lubridate::as.period(round(d_mu-2.33*d_sd)),
                       ifelse(d_mu-2.33*d_sd<0, " (truncated to 0)", ""),
                       "\n99%: ", lubridate::as.period(round(d_mu+2.33*d_sd)))
        )
    })
    
    last_save_location = reactiveVal("")
    observeEvent(input$save_annotations, {
      last_save_location(paste0(ann_dir,
                                "/annotations", 
                                ifelse(input$save_annotation_suffix=="",
                                       "",
                                       paste0("_",input$save_annotation_suffix)
                                ),
                                ".rds.gz"))
      all_note_annotations() %>% write_rds(last_save_location())
    })
    output$save_annotation_response = renderText({
      last_save_location()
    })
    
    output$plot <- renderPlot({
      dat <- curpt()
      outcomes = all_outcomes()
      features = all_features()
      inclusions = all_inclusions()
      exclusions = all_exclusions()
      originaldat = zoom_pts()[[length(zoom_pts())]]
      ninclusions = originaldat %>% filter(event %in% (inclusions %>% .[,1] %>% .[[1]])) %>%
        select(event) %>% distinct() %>% nrow()
      nexclusions = originaldat %>% filter(event %in% (exclusions %>% .[,1] %>% .[[1]])) %>%
        select(event) %>% distinct() %>% nrow()
      
      bp = brushedPoints(dat, input$plot_brush)
      if(!is.null(bp)) {
        brush_lb = lubridate::as_datetime(input$plot_brush$xmin)
        brush_ub = lubridate::as_datetime(input$plot_brush$xmax)
        inex = paste0(brush_lb, " --", brush_ub)
      }
      
      dat = dat %>%
        mutate(Outcome=event %in% (outcomes[,1] %>% .[[1]])) %>%
        mutate(Event=factor(pmin(-3*Outcome,-as.numeric(event %in% (features[,1] %>% .[[1]]))),
                              levels=c(-3, -1, 0)) %>%
                 forcats::fct_recode(Outcome="-3",Feature="-1",Data="0") %>%
                 forcats::fct_relevel(c("Data","Outcome","Feature"))) %>%
        arrange(Event)
      if(yvar() == "auto") {
        dat = dat %>% left_join(auto_map(), by="eventi") %>% 
          mutate(Assignment = forcats::fct_explicit_na(Assignment, na_level="Group NA")) %>%
          mutate(auto=ifelse(is.na(combined),-1,combined))
      }
      
      coloring = ifelse(input$event_color=="Group" & input$event_order=="Auto","Assignment","Event")
      
      pc <- ggplot(dat, aes_string(xvar(), yvar(), size="Outcome", color=coloring)) +
        geom_point() +
        ggthemes::theme_economist_white(horizontal=F) +
        scale_color_colorblind(drop=F) +
        scale_size_discrete(guide='none') +
        scale_x_datetime(breaks=scales::breaks_pretty(n=20),guide=guide_axis(angle=90)) + 
        ylab("Events") +
        annotation_custom(grob=grid::textGrob(inex, x=0.96, y=0.96, just="right"))

      if(input$lb_zoom_out_exponent!=0 | input$ub_zoom_out_exponent!=0) {
        lbz = input$lb_zoom_out_exponent
        ubz = input$ub_zoom_out_exponent
        mintime = min(dat$TIME,na.rm=T)
        maxtime = max(dat$TIME,na.rm=T)
        duration = maxtime-mintime
        pc = pc +
          xlim(mintime - (2**lbz)*duration*0.5, maxtime + (2**ubz)*duration*0.5)
      }
      
      if(yvar()=="auto" && input$ncenters>7) {
        crp = colorRampPalette(c("black", "green", "blue", "orange","purple", "red", "gray","yellow"))
        pc = pc + scale_color_manual(values=crp(input$ncenters+1))
      }
      
      # Add line segment of bounds
      currange = curlbub() %>% as.POSIXct(origin="1970-01-01")
      if(!is.null(currange) && (is.infinite(currange) %>% sum() != 2)) {
        pc_ybounds = ggplot_build(pc)$layout$panel_scales_y[[1]]$range$range
        pc = pc + geom_segment(y=pc_ybounds[[1]], yend=pc_ybounds[[1]], x=currange[1], xend=currange[2],
                               size=0.5, show.legend=F, color="black")
      }
      
      pc
    })
    output$plot_hoverinfo <- renderPrint({
      cat("input$plot_hover:\n")
      str(input$plot_hover)
    })
    output$plot_brushinfo <- renderPrint({
      cat("input$plot_brush:\n")
      str(input$plot_brush)
    })
    output$plot_brushed_points <- DT::renderDataTable({
      dat <- curpt()
      if(yvar() == "auto") {
        dat = dat %>% left_join(auto_map(), by="eventi") %>% 
          mutate(Assignment = forcats::fct_explicit_na(Assignment, na_level="Group NA")) %>%
          mutate(auto=ifelse(is.na(combined),-1,combined))
        res <- brushedPoints(dat, input$plot_brush) %>% select(-auto,-Assignment, -combined)
      } else {
        res <- brushedPoints(dat, input$plot_brush)  
      }
      
      datatable(res)
    })

    
    output$annotate_plot_hoverinfo <- renderPrint({
      cat("input$annotate_plot_hover:\n")
      str(input$annotate_plot_hover)
    })
    output$annotate_plot_brushinfo <- renderPrint({
      cat("input$annotate_plot_brush:\n")
      str(input$annotate_plot_brush)
    })
    
    
    observeEvent(input$note_annotations_remove, {
      all_note_annotations(
        all_note_annotations()[-input$note_annotations_rows_selected,]
      )
    })
    
    observeEvent(input$note_annotations_remove_all, {
      all_note_annotations(blank_annotation())
    })
    
    observeEvent(input$note_annotations_clear_selection, {
      dt = DT::dataTableProxy("note_annotations",session=session)
      dt %>% selectRows(selected = NULL)
    })
    
    output$note_annotations <- DT::renderDataTable({
      dat <- curpt()
      datatable(all_note_annotations() %>% 
                  filter(pt == as.character(dat$hosp_id[1])) %>%
                  select(-bounds, -text_position, -selection_reasoning)
      )  # only show this patient's annotations
    })
    
    observeEvent(c(input$slider_lb_uncertainty, input$annotation_lbub_tied), {
      if(input$annotation_lbub_tied=="On"& input$slider_ub_uncertainty!=input$slider_lb_uncertainty) {
        updateSliderTextInput(session=session, "slider_ub_uncertainty",selected=input$slider_lb_uncertainty) 
      }
    })
    observeEvent(input$slider_ub_uncertainty, {
      if(input$annotation_lbub_tied=="On") {
        updateSliderTextInput(session=session, "slider_lb_uncertainty",selected=input$slider_ub_uncertainty) 
      }
    })
    
    blank_annotation = reactiveVal(NULL)
    observe({  # run on init only
      blank_annotation(
        tibble(text="",
               text_position=tibble(start=1,end=1),
               selection_reasoning=list(tibble(term="")),
               bounds = list(tibble(a=1:2)),
               pt = isolate(input$pts),
               time=lubridate::now(),
               negated=F,
               annotator=input$save_annotator) %>%
          .[0,]
      )
    })
    
    observeEvent(c(input$record_annotation,kb_record_annotation),{
      print(kb_record_annotation())
      
      ana = all_note_annotations()
      if(is.null(ana)) {
        if(is.null(seed_annotation_file)) {
          all_note_annotations(blank_annotation())
        } else {
          anns = read_rds(seed_annotation_file)
          all_note_annotations(anns)
        }
        return(NULL)
      }
      if (!is.null(input$plot_brush)) {
        # browser()
        if("numeric" %in% class(curpt()$TIME[[1]])) {
            brush_lb = lubridate::as_datetime(input$plot_brush$xmin)
            brush_ub = lubridate::as_datetime(input$plot_brush$xmax)
        } else {
            brush_lb = input$plot_brush$xmin
            brush_ub = input$plot_brush$xmax
        }
        if(!input$annotation_lb_use) {
          brush_lb = -Inf
        }
        if(!input$annotation_ub_use) {
          brush_ub = Inf
        }
      } else {
        warning("No interval selected, so no annotation made")
        return(NULL)
      }
      
      if(input$annotation_bound_mode == "Bounds") {
        bounds = list(lbub=tibble(lb=brush_lb, ub=brush_ub))
      } else {  # Probability
        lb_sd = as.numeric(lubridate::as.duration(input$slider_lb_uncertainty))
        ub_sd = as.numeric(lubridate::as.duration(input$slider_ub_uncertainty))
        dur_sd = as.numeric(lubridate::as.duration(input$slider_duration_uncertainty))
        
        bounds = list(lbub_sds=
                        list(lbub=tibble(lb=brush_lb, ub=brush_ub),
                             sd = 
                               tibble(lb=lb_sd, 
                                      ub=ub_sd, 
                                      dur=dur_sd)
                        )
        )
      }
      
      annotation = 
        tibble(text=input$selected_text$text,
               text_position=tibble(start = input$selected_text$start, end=input$selected_text$end),
               selection_reasoning=list(tibble(term=input$plot_brushed_points_search,
                                               rows_selected=
                                                 list(brushedPoints(curpt(),input$plot_brush) %>%
                                                        .[input$plot_brushed_points_rows_selected,])
               )),
               bounds = bounds,
               negated = input$annotation_selection_negation,
               pt = input$pts,
               time=lubridate::now(),
               annotator=input$save_annotator)
      
      all_note_annotations(
        all_note_annotations() %>%
          bind_rows(
            annotation
          )
      )
    })
    
    output$annotations_chosen <- DT::renderDataTable({
      datatable(all_annotations(),
                options=list(dom="lt", compact=T,
                             pageLength=10,
                             lengthMenu = list(c(10, -1), c("10", "All"))),
                autoHideNavigation=T, selection="multiple")
    })

    # combine selected events (via dropdown and from graph)
    efp = reactive({
      ve = input$varinspect_event
      if(input$varinspect_drag_add) {
        se = selected_events()
      } else {
        se = NULL
      }
      if(!is.null(se)) {
        if(is.null(ve)) {
          eventoi = se$event
        } else {
          eventoi = c(ve, se$event)
        }
      } else {
        eventoi = ve
      }
      return(eventoi)
    })
    efp_view_limit=reactive({10})

    # Plot the value graphs
    output$varinspectui = renderUI({
      ns = session$ns
      efp = efp() %>% head(efp_view_limit())
      validate(need(!is.null(efp) || length(efp)>0, "Select an event for plotting values"))
      
      plotOutput(ns("varinspect_plot"), height=200*length(efp), width="100%")
    })
    output$varannotateui = renderUI({
      ns = session$ns
      plotOutput(ns("varannotate_plot"), height=200*length(efp), width="100%",
                 hover = hoverOpts(
                   id = ns("annotate_plot_hover")
                 ),
                 brush = brushOpts(
                   id = ns("annotate_plot_brush"),
                   resetOnNew = F  # ns("brush_reset")
                 )
      )
    })
    output$varinspect_plot = renderPlot({
      local_curpt = curpt()
      eventoi = efp() %>% head(efp_view_limit())

      plottingDat = local_curpt %>% select(TIME, event, value) %>% filter(event %in% eventoi) %>%
        distinct()
      usingAll = F
      if (nrow(plottingDat)==0) {
        usingAll = T
      }
      if(class(plottingDat$TIME)[[1]]!="numeric") {
        plottingDat = plottingDat %>%
          mutate(TIME=as.POSIXct(TIME))
      }
      
      df_subplots = plottingDat %>% group_by(event) %>% 
        mutate(naCount = is.na(as.numeric(value))) %>%
        summarise(fraction=sum(naCount)/n()) %>%
        mutate(treat_as_numeric=fraction<0.6)
      
      gp = plottingDat %>%
        filter(event %in% (df_subplots %>% filter(treat_as_numeric) %>% .[["event"]])) %>%
        mutate(value=as.numeric(value))
      # browser()
      if(nrow(gp)>0) {
        gp = gp %>%
          ggplot(data=., aes(y=value, x=TIME)) +
          geom_point() +
          facet_wrap(~event, ncol=1, scales="free") +
          ylab("Value") + xlab("") +
          scale_x_datetime(breaks=scales::breaks_pretty(n=20),guide=guide_axis(angle=90)) +
          theme_economist_white() +
          scale_color_colorblind(drop=F)
        if (!is.null(input$plot_brush)) {
          if("numeric" %in% class(local_curpt$TIME[[1]]) ||
             "POSIXct" %in% class(local_curpt$TIME[[1]])
             ) {
            brush_lb = lubridate::as_datetime(input$plot_brush$xmin)
            brush_ub = lubridate::as_datetime(input$plot_brush$xmax)
          } else {
            brush_lb = input$plot_brush$xmin
            brush_ub = input$plot_brush$xmax
          }
          gp = gp + annotate("rect", xmin=brush_lb, xmax=brush_ub, ymin=0, ymax=Inf, alpha=0.25, fill = "#99ccff")
        }
        if(!("numeric" %in% class(local_curpt$TIME[[1]]))) {
          gp = gp + 
            coord_cartesian(xlim=c(min(local_curpt$TIME) %>% as.POSIXct(),
                                   max(local_curpt$TIME) %>% as.POSIXct()))
        } else {
          if(nrow(local_curpt)>1) {
            gp = gp + coord_cartesian(xlim=c(min(local_curpt$TIME), max(local_curpt$TIME)))
          }
        }
      } else {
        gp = NULL
      }
      
      gp2 = plottingDat %>%
        filter(!event %in% (df_subplots %>% filter(treat_as_numeric) %>% .[["event"]])) %>%
        mutate(value=forcats::fct_explicit_na(as.factor(value)))
      if(nrow(gp2)>0) { 
        gp2 = ggplot(data=gp2, aes(y=value, x=TIME)) +
          geom_point() +
          facet_wrap(~event, ncol=1, scales="free") +
          ylab("Value") +
          theme_economist_white() +
          scale_color_colorblind(drop=F)
        if(!("numeric" %in% class(local_curpt$TIME[[1]]))) {
          gp2 = gp2 + xlim(as.POSIXct(min(local_curpt$TIME),origin = "1970-01-01"),
                           as.POSIXct(max(local_curpt$TIME),origin = "1970-01-01"))
        } else {
          if(nrow(local_curpt)>1) {
            gp2 = gp2 + coord_cartesian(xlim=c(min(local_curpt$TIME), max(local_curpt$TIME)))
          }
        }
      } else {
        gp2 = NULL
      }
      if(is.null(gp2)) {
        plot(gp)
      } else if (is.null(gp)) {
        plot(gp2)
      } else {
        plot(grid.arrange(gp, gp2, ncol=1))
      }
    })
    
    
    output$varannotate_plot = renderPlot({
      if(nrow(all_note_annotations())==0) { return(NULL) }
      local_curpt = curpt()
      eventoi = efp() %>% head(efp_view_limit())

      plottingRange = local_curpt %>% select(TIME, event) %>% filter(event %in% eventoi) %>%
        summarise(time.min = min(TIME, na.rm=T), time.max=max(TIME, na.rm=T))
      usingAll = F
      if (!(plottingRange %>% map(is.finite) %>% c() %>% all())) {
        usingAll = T
        plottingRange = local_curpt %>% select(TIME) %>%
          summarise(time.min = min(TIME, na.rm=T), time.max=max(TIME, na.rm=T))
      }
      if(class(plottingRange$time.min)[[1]]!="numeric") {
        plottingRange = plottingRange %>% mutate_all(as.POSIXct)
      }

      plottingRange = local_curpt %>% select(TIME, event) %>% filter(event %in% eventoi) %>%
        summarise(time.min = min(TIME, na.rm=T), time.max=max(TIME, na.rm=T))
      usingAll = F
      if (!(plottingRange %>% map(is.finite) %>% c() %>% all())) {
        usingAll = T
        plottingRange = local_curpt %>% select(TIME) %>%
          summarise(time.min = min(TIME, na.rm=T), time.max=max(TIME, na.rm=T))
      }
      if(class(plottingRange$time.min)[[1]]!="numeric") {
        plottingRange = plottingRange %>% mutate_all(as.POSIXct)
      }
      
      if(input$lb_zoom_out_exponent!=0 | input$ub_zoom_out_exponent!=0) {
        lbz = input$lb_zoom_out_exponent
        ubz = input$ub_zoom_out_exponent
        duration = plottingRange$time.max-plottingRange$time.min
        # Overwrite the plottingRange here
        plottingRange$time.max = plottingRange$time.max + (2**ubz)*duration*0.5
        plottingRange$time.min = plottingRange$time.min - (2**lbz)*duration*0.5
      }
      
      note_annotations_rows_selected = input$note_annotations_rows_selected
      if(is.null(note_annotations_rows_selected)) {
        note_annotations_rows_selected = -1
      }
      
      dat = all_note_annotations() %>% 
        filter(pt == as.character(local_curpt$hosp_id[1])) %>% 
        select(bounds, text_position) %>% 
        mutate(bounds = modify_if(bounds, ~ !is_tibble(.x), ~ .x[[1]])) %>% 
        unnest(everything()) %>%
        mutate(selected = seq(1,n(),length.out=n()) %in% note_annotations_rows_selected)
        
      if(nrow(dat)==0) { return(NULL) }
      dat = dat %>%
        select(interval.lb=lb, interval.ub=ub, selected) %>% 
        mutate(interval.lb = lubridate::as_datetime(interval.lb),
               interval.ub = lubridate::as_datetime(interval.ub)) %>% 
        mutate(dur=interval.ub-interval.lb) %>%
        mutate(is.interval=T) %>%
        mutate(duration = lubridate::time_length(dur, "hours"))
      
      ggplot(data = dat) +
        geom_segment(aes(x=interval.lb, xend=interval.ub, y=duration, yend=duration,
                         color=selected), alpha=0.5) +
        ggthemes::theme_economist_white(horizontal=F) +
        scale_color_colorblind(drop=F) +
        coord_cartesian(xlim=c(plottingRange$time.min, plottingRange$time.max)) +
        xlab("") + ylab("Duration (hours)") +
        scale_y_log10()
    })
    
    return(list(csvFileLoader=csvFileLoader))
  }
  )
}
