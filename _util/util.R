library(ggplot2)

simon_theme <- function()
{
 theme(axis.line = element_line(colour = 'black', size = .25),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),          
          axis.ticks.y=element_blank(),
          axis.title=element_text(size=10),
          axis.text=element_text(size=10),
          axis.ticks.x=element_blank())
    
}