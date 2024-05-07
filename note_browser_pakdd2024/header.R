get_header = function() {
  return("pre, table.table {
            font-size: smaller;
          }
                      
          body {
            min-height: 2000px;
          }
          
          #cohort-note_contents {
            overflow-y: auto;
            height: 750px;
          }
                      
          .option-group {
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 4px 5px;
            margin: 1px 0px;
            background-color: #f5f5f5;
            # background-color: #e5e5e5;
          }
              
          .option-group-inline {
            display: inline-block;
            padding: 0px 5px;
            margin-bottom: 10px;
          }
                    
          .option-group-inline-top {
            vertical-align: top;
          }
          
          .option-group-inline-bottom {
            vertical-align: bottom;
          }
                    
          .option-group-inline-right {
            margin-right: 5px;
          }
                      
          .option-group-inline-left {
            margin-right: 2px;
          }
                    
          .option-header {
            color: #79d;
            text-transform: uppercase;
            margin-bottom: 5px;
          }"
  )
}