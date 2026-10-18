This post will be the most research intense post we've ever done. You will need to run many code simulations, derive math formulas, and plan ahead, before doing any writing. Start by reading this paper https://arxiv.org/pdf/2102.11297 which introduces the idea of data compression and sufficient statistics. After you read this I will guide you on what we will write. 

Get ready to listen first, and then plan, design, research, and lastly do the writing. The major concept we will address is how to run linear regression in an extremely efficient way. We will do this by leveraging the structure of the regression form, knowing that certain parameters are important for causal inference, and certain parameters are not. The trickiness of this research is that the compression strategy ought to be robust to multiple types of input data, namely randomized AB tests, as well as cluster randomized tests, as well as encouragement experiments as well as adaptive tests. These 4 differ in the sense:

1. A completely randomized AB test has independent data, which can make the covariance of the error term simple.
2. Cluster randomized tests have dependent data. There will be within-user correlation that needs to be accounted for, which will lead to the need for cluster robust standard errors on the treatment effect. Within this class of experiments, there are special forms that are very practical and interesting. Say the data comes in the form of a panel: y_{i, j} with covariates X_{i, j} and randomization T determined at the i level. There are special cases where X_{i, j} are also determined at the i level, which can make the data compression strategy simpler as all rows j within i have the same covariates X_i. There are also special cases where the panel is a balanced panel.
3. In an encouragement test, we have a randomized instrumental variable, and a endogeneous variable of interest. We will need to generalize from linear regression to two stage least squares, and find a compression strategy. In addition, this type of test can operate on independent data, as well as cluster randomized data. Other than pivoting to two stage least squares, we will need to think about two stage least squares with HC1 robust standard errors, versus two stage least squares with cluster robust standard errors.
4. In an adaptive experiment, we run the experiment for Q days, but with differing traffic split rates. Each day brings a new cohort of users that have first entered the experiment. On day 1 the traffic split rates splits users into treatment and control with propensity p1. On day 2, it splits with propensity p2, etc. This will feel like an AB test, but it has propensity weights.

These are 4 different types of experiments that produce different forms of the model. Additionally, there are different types of metrics that we want to model, some are regular metrics and some are ratio metrics. We have written about ratio metrics before, which can be thought of as a type of bivariate linear regression, with an extra modeling step to combine inference from the numerator and denominator to get the variance of the ratio. Pay attention to all these different forms that we want to keep track of. 

To summarize, think about creating a structure for:
1. The type of experiment
2. The type of metric.
3. The covariance type, which might come from the experiment type. It is either HC1 robust standard errors or cluster robust standard errors.

I have a theory that across these 4 types of experiments, and these 2 different types of metrics, and these 2 different types of standard errors, there is a way to generalize all of these model specifications into a single umbrella. My theory is that as long as we can create a very efficient implementation for 2 stage least squares with cluster robust standard errors and with propensity weights, suitable for ratio metrics, then everything else is just a special case. There is some writing for this theory here https://arxiv.org/pdf/2604.24116. Make sure to read this.

Much of the original YOCO (you only compress once) paper was written with a very specific regression form in mind. It was a non-ratio metric, regressed on the treatment dummy with categorical X covariates. A compression strategy was derived from that structure, and it can work across all 3 experiment types. However, it is very limiting. Many users will want to regress in this form

y regressed on treatment dummy interacted with X covariates, where X can be categorical or it can be continuous.

CUPED is a form of this, where X is just a single continuous number. It is not clear to me how to extend the YOCO framework to have continuous covariates. However, we have derived approximations here https://jeffwong.github.io/posts/robust_cuped/.

Other than training the linear model, we will need a strategy that takes the coefficients, the covariance matrix of the parameters, and then combines them correctly to get the treatment effect and its variance. This has already been discussed as the Delta Vector Strategy https://arxiv.org/pdf/2412.08788. The strategy here exploits the fact that some coefficients in the model are useful for inference, and some are not. As a result, there is a computational trick for how to efficiently select the ones that matter, and then combine them. However, this strategy was written as an afterthought, and it might be possible to leverage this separation of parameters that are important for inference, and parameters that are not important, earlier. Your research will need to study how to best leverage this.

Our task is going to be very ambitious. First, we need to read all the links I provided. Next, we need to research and run code simulations. You will need to derive new math formulas. Finally, we will write. Our research and simulations should come up with a single umbrella view that can handle

1. The four types of experiments listed above
2. The 2 types of metrics
3. The 2 types of covariance structures
4. The form of the linear model with interaction terms.

An ideal outcome is to show that a single implementation of bivariate two stage least squares run on clustered randomized data with cluster robust standard errors and propensity weights covers all combinations of these model specifications. The ideal outcome will create a data compression strategy, similar to conditionally sufficient statistics, that will shrink large volumes of data to smaller data frames, leveraging rows of the data that are similar or even identical to one another. It will understand that in a panel data, it's not only rows that need to be identical, because we need to track all within-user information. The research will optionally leverage extra properties, for example cases of balanced panels, or other structure that we can exploit. It will implement both the compression strategy, as well as a unique method for how to train the two stage least squares model, and then finally it will use Delta Vectors, or similar strategy, to combine the right coefficients and right covariances to get at the treatment effect and its variance. If possible, you ought to invent a way to use the delta vector strategy earlier to partition the important parameters and the not-important parameters. Consider other formulas like Frisch Waugh Lovell, or DoubleML.

If the ideal outcome cannot be achieved, we will find the best second order approximation. We can also consider rounding the continuous X covariates to increase the amount of overlap, as long as we justify that the transformation on the X covariate is done pre treatment and is exogeneous.

Again start with thinking, research, and design. Demonstrate to yourself, and to me, that a single model can truly cover all cases. Form your own synthetic data and use it as a recurring example. We can turn your research into an appendix. Then, as the final step, do your writing.


---

This is a very good milestone for the post. However, it still needs several major revisions. You will need to rewrite entire sections and even delete entire sections. Again, listen first, research, run code simulations, and then at the last step do the writing.

Your post assumes that the reader has read the You Only Compress Once paper. Do not assume they have read it. Reintroduce the major concepts from there. It is not necessary to cite it, and it is also not necessary to keep using their notation. It is not necessary to create criticism of that paper.

The ordering of your text needs to change. It begins with a categorical segment, because You Only Compress Once starts that way. Also, you are not making use of the categorical segment as a covariate for variance reduction, you are esentially just doing the naive t test and wald test. We want to make sure to present the more general OLS and 2SLS immediately, and bypass the naive ones. You waited all the way until the end to expand to continuous covariates. Do not tie yourself to the structure of the You Only Compress Once paper. You should follow this logical ordering below instead.

Introduce a recurring example, just as you have done. However, we need to make it more convincing that we want to model the panel at the y_{c,j} level. Your example prompts the question - why not just preaggregate at the beginning since the treatment effect question is about members? We need an example where we want to model at the (c,j) level and we would not want to preaggregate. 

Next, Explain how it is a cluster randomized encouragement experiment with an IV and propensity weights and ratio metrics. Then, without thinking about compression, explain how a bivariate two stage least squares with robust clustered standard errors and propensity weights can model this experiment. Then, explain how this model is a universal model. It can capture purely randomized AB tests, it can capture non-ratio metrics, etc. Think about all the permutations of experiment specifications and produce a summary table showing this is a universal model that can handle all permutations. In the appendix show the mathematical proof that each permutation is a special case of this universal model.

After showing this universal model we will derive the compression strategy from first principles. Instead of presenting the categorical-only covariate case first, start by discussing the interaction model, where X can either be continuous or categorical. Be prepared that there are multiple covariates. Note that in the case of multiple covariates, it is not necessary to model the full interaction model e.g.

y ~ T * X1 + T * X2 + T * X1 * X2, where this model has every pairwise interaction possible.

We would only ask for an interaction model where all covariates are interacted with T, but covariates need not interact with each other, e.g.

y ~ T*X1 + T*X2

 With this in mind, first write down the strategy for the special case of a single covariate (This should look like CUPED), as well as the special case of only categorical covariates (which looks like You Only Compress Once, but don't reference that paper - derive it again from first principles). Then finally, present a strategy for compression when both categorical and continuous covariates are present. It may be necessary to make second order approximations, or bin the covariates, or cluster the continuous data. Do research to determine which strategies are good and present the best strategy for multiple continuous and categorical covariates.

 A good outcome is that the post is much shorter. It should not contain the baggage of the You Only Compress Once paper. It should not have to criticize the mistakes. IT should not follow the notation. It should not follow its logical order. This new post should simply claim 1 universal model can cover all of our cases, and there is a compression strategy for that universal model. It will include R and Rcpp code to implement that compression strategy and implement code for training the model, using delta vectors to efficiently extract the important parameters, and then compute the treatment effect and its variance. It will benchmark for correctness using textbook code that doesn't implement compression.

 ---

 This is an interesting post and strategy. However, we will need to make several editorial revisions in order to improve clarity. Your choice of notation is not very clear, and you need to display more intermediate print outs of what data frames and matrices you are talking about. As examples

 1. You describe a dataset of members and profiles, with covariates, streams and rebuffers. But you never printed an example of such a dataframe. This should be near the top of the post.
 2. Your model references to types of matrices, M and Q. You should print an example of what the rows of these matrices are.
 
 For your mathematical notation, you need to choose notation that is more clear.

 1. You use c for members. This doesn't make sense. Choose either m for members, or a for accounts. You use j to index profiles, this ought to be p.

 2. Your choice of matrices M and Q doesn't make sense. A better choice could be M_1 for the matrix for the first stage, and M_2 for the matrix for the second stage.

 3. You introduce a product, A = Q^T W M. I don't even know if this has physical meaning. If it is simply shorthand to make formulas shorter, say so, and introduce it earlier as a convenient shorthand.

 4. You use eta_c as something that looks like a residual. Again you haven't even stated what the first stage and second stages are, so it is inapproproiate to use this with its physical meaning without stating the model. Also, do not use eta, use something like epsilon or varepsilon. 

 5. You introduced r_{c,j} which is a subset of the covariate matrix. It has a "reference" which feels like demeaning the columns of the covariate matrix. Say so and include an intermediate printout of what this matrix is.

 6. In your model you have main effects for the demeaned covariates, and the interaction terms. Are you missing the main effect on T itself? 

 ---

 The section "Which effect the model estimates" lacks a lot of clarity. You are talking about partialing r out of instruments Zr, and then creating the moment conditions. This is too complicated and the description should not be necessary. It is probably confusing because the logical ordering is wrong.

 We first lead the post using the running example. That is correct. Then we say that the design actually has four experiment specifications in it, the features, which is going to make the analysis of this experiment difficult. This is also the correct way to order things.

 The next ordering is not correct. Instead of introducing the covariates, then the model, then the effects the model can estimate, we need to weave these all together in one section.

 You go immediately into covariates and technical details of the model. Instead, we should slowly introduce the two stage least squares model first, state its form. Then, state its physical interpretation. Show the reader that the physical interpretation is connected to the recurring example, but it is also a universal model that can answer many other types of effects for other types of experiment design. This justification of the model should appear immediately after the model is introduced. Then, we will describe how the model is going to do that, which corresponds to your technical details of the covariates, and how the matrices are multiplied and how the covariance is derived. This narrative will follow a "Problem -> What Solution -> Why -> How" logical ordering. Your narrative is a "Problem -> How -> What solution and why" ordering.  

 ---

 The "Compressing the Universal Model" section is very dense. We need to weave more code examples and intermediate print outs in. For the code examples, use dplyr to manipulate the (account, profile) level dataframe into the compressed records. First do this on "base cases" that are easy to understand, e.g. 1 continuous covariate as in CUPED, and then another case of 1 categorical covariate. Include the intermediate print outs so that the reader can build an intuition of the data transformations that map to your mathematical formulas. Then, present the generalized data compression function for the case of many covariates, some categorical and some continuous.

 Your section on "using the delta vector early" is also dense. It is not clear to me whether there is an optimization at all. Normally, we would train the model, obtain the fitted coefficients and covariances, and then combine the important subsets using the delta vector. Think carefully about whether an optimization exists, and similarly explain it slowly with intermediate print outs. You should have one implementation that only uses the delta vectors at the end without optimization, and if reasonable, a second implementation that optimizes the algorithm.