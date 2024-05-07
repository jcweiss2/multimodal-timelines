require(shiny)
require(data.table)
require(tidyverse)
require(DT)
require(bsplus)

listModuleUI <- function(id, heading = "Generic List", keyterm="list") {
  ns <- NS(id)
  bs_delay = list(show = 1000, hide = 50)
  
  tagList(
    div(class = "option-group",
        div(class = "option-header", heading),
        checkboxInput(ns("list_show"),paste("Show",keyterm,"criteria")),
        conditionalPanel("input.list_show",ns=ns,
                         DT::dataTableOutput(ns("chosen"))),
        textInput(ns("regexmodify"), label=NULL, width="70%") %>%
          bs_embed_tooltip(title = "Add/remove event by regular expression", delay=bs_delay),
        div(class = "option-group-inline",actionButton(ns("add"),"Add")),
        div(class = "option-group-inline",actionButton(ns("remove"),"Remove")) #,
    )
  )
}

event_selection = function(list_of_events, list_to_update, selected_events=(function() {return(NULL)})) {
  return(
    function(input, output, session) {
      observeEvent(input$add,{
        ltext = isolate(input$regexmodify)
        if(ltext != "") {
          events = list_of_events()
          list_to_update(list_to_update() %>%
                           bind_rows(events %>% filter(str_detect(events[,1] %>% .[[1]], pattern=ltext))) %>%
                           distinct()
          )
        } else {
          if(!is.null(selected_events())) {
            list_to_update(list_to_update() %>%
                             bind_rows(selected_events() %>% select(event)) %>%
                             distinct()
            )
          }
        }
      })
      
      output$chosen <- DT::renderDataTable({
        datatable(list_to_update(), 
                  options=list(dom="lt", compact=T,
                               pageLength=10,
                               lengthMenu = list(c(10, -1), c("10", "All"))),
                  autoHideNavigation=T, selection="multiple")
      })
      
      observeEvent(input$remove,{
        ltext = isolate(input$regexmodify)
        rows_to_remove = isolate(input$chosen_rows_selected)
        events = list_to_update()
        if(!is.null(rows_to_remove) && length(rows_to_remove)>0) {
          list_to_update(list_to_update() %>% slice(-rows_to_remove))
        } else {
          if(ltext != "") {
            list_to_update(events %>% filter(!str_detect(events[,1] %>% .[[1]], pattern=ltext)))
          } else {
            if(!is.null(selected_events())) {
              list_to_update(events %>%
                               filter(!(event %in% 
                                          (selected_events() %>% select(event) %>% distinct() %>% .[["event"]])
                               ))
              )
            }
          }
        }
      })
      
      
    }
  )
}
  
listModuleServer <- function(id, f) {
  moduleServer(
    id,
    f
  )
}