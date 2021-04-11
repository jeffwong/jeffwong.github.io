# Tech Stack

This blog is built using the wonderful packages
[RMarkdown](https://rmarkdown.rstudio.com/) and [Distill](https://pkgs.rstudio.com/distill/). The blog is hosted on [github pages](https://pages.github.com/).

# Contributing

To contribute a post, add a distill post to the `_posts` directory by running `distill::create_post("Name of Post")`, then knit the post using RStudio, or `rmarkdown::render("<path to this repo>/_posts/<path to the RMarkdown file>")`. To combine the blog with the post, run `rmarkdown::render_site()`. Finally, use git to commit the new post and push it to the repository.