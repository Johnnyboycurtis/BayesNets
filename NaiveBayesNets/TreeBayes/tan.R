## tree augmented naive bayes code


# naive Bayes frontend.
naive.bayes = function(x, training, explanatory) {

  bayesian.classifier(x, training = training, explanatory = explanatory,
    method = "naive.bayes", whitelist = NULL, blacklist = NULL, expand = list(),
    debug = FALSE)

}#NAIVE.BAYES

# tree-augmented naive Bayes frontend.
tree.bayes = function(x, training, explanatory, whitelist = NULL, blacklist = NULL,
    mi = NULL, root = NULL, debug = FALSE) {

  bayesian.classifier(x, training = training, explanatory = explanatory,
    method = "tree.bayes", whitelist = whitelist, blacklist = blacklist,
    expand = list(estimator = mi, root = root), debug = debug)

}#TREE.BAYES



# baeysian network classifiers.
bayesian.classifier = function(data, method, training, explanatory, whitelist,
    blacklist, expand, debug = FALSE) {

  # check debug.
  check.logical(debug)
  # check the learning algorithm.
  check.learning.algorithm(method, class = "classifier")
  # check the training node (the center of the star-shaped graph).
  check.nodes(training, max.nodes = 1)
  # check the data.
  check.data(data, allowed.types = discrete.data.types)

  # check the explantory variables.
  if (missing(data)) {

    check.nodes(explanatory)

  }#THEN
  else {

    vars = names(data)
    # check the label of the training variable.
    check.nodes(training, graph = vars, max.nodes = 1)
    # check the labels of the explanatory variables.
    if (missing(explanatory))
      explanatory = vars[vars != training]
    else
      check.nodes(explanatory, graph = explanatory)

  }#ELSE

  # check that the training node is not included among the explanatory variables.
  if (training %in% explanatory)
    stop("node ", training, " is included in the model both as a training ",
         "and an explanatory variable.")
  # cache the whole node set.
  nodes = c(training, explanatory)
  # sanitize whitelist and blacklist, if any.
  if (method != "naive.bayes") {

    whitelist = build.whitelist(whitelist, nodes = nodes, data = data,
                  algo = method, criterion = "mi")
    blacklist = build.blacklist(blacklist, whitelist, nodes, algo = method)

    if (method == "tree.bayes") {

      # arcs to and from the training node cannot be whitelisted or blacklisted.
      if ((training %in% whitelist) || (training %in% blacklist))
        stop("blacklisting arcs to and from the training node is not allowed.")

    }#THEN

  }#THEN

  # sanitize method-specific arguments.
  extra.args = check.classifier.args(method = method, data = data, extra.args = expand,
                 training = training, explanatory = explanatory)

  if (method == "naive.bayes") {

    # naive bayes requires no test.
    ntests = 0
    # not test statistiic involved.
    test = "none"

    res = naive.bayes.backend(data = data, training = training,
            explanatory = explanatory)

  }#THEN
  else if (method == "tree.bayes") {

    # tan gets its tests from the chow-liu algorithm.
    ntests = length(explanatory) * (length(explanatory) - 1)/2
    # same for the test
    test = as.character(mi.estimator.tests[extra.args$estimator])

    res = tan.backend(data = data, training = training, explanatory = explanatory,
            whitelist = whitelist, blacklist = blacklist, mi = extra.args$estimator,
            root = extra.args$root, debug = debug)

  }#THEN

  # set the learning algorithm.
  res$learning$algo = method
  # set the metadata of the network in one stroke.
  res$learning = list(whitelist = whitelist, blacklist = blacklist,
    test = test, ntests = ntests, algo = method, args = extra.args)
  # set the trainign variable, for use by predict() & co.
  res$learning$args$training = training

  invisible(res)

}#BAYESIAN.CLASSIFIER




# backend of the naive Bayes classifier.
naive.bayes.backend = function(data, training, explanatory) {

  # cache the node set.
  nodes = c(training, explanatory)
  # create the empty graph.
  res = empty.graph(nodes)
  # create the set of arcs outgoing from the training variable.
  res$arcs = matrix(c(rep(training, length(explanatory)), explanatory),
               ncol = 2, byrow = FALSE)
  # update the network structure.
  res$nodes = cache.structure(nodes, arcs = res$arcs)
  # set a second class "bn.naive" to reroute method dispatch as needed.
  class(res) = c("bn.naive", "bn")

  return(res)

}#NAIVE.BAYES.BACKEND





# backend of the TAN algorithm.
tan.backend = function(data, training, explanatory, whitelist, blacklist,
    mi, root, debug) {

  # set a dummy estimator variable.
  estimator = 1L
  # cache the node set.
  nodes = c(training, explanatory)
  # create the empty graph.
  res = empty.graph(nodes)
  # create the set of arcs outgoing from the training variable.
  class.arcs = matrix(c(rep(training, length(explanatory)), explanatory),
               ncol = 2, byrow = FALSE)

  # call chow-liu to build the rest of the network.
  chow.liu.arcs = chow.liu.backend(x = minimal.data.frame.column(data, explanatory),
                    nodes = explanatory, estimator = estimator,
                    whitelist = whitelist, blacklist = blacklist,
                    conditional = minimal.data.frame.column(data, training, drop = TRUE),
                    debug = debug)

  # set the directions of the arcs in the Chow-Liu tree.
  chow.liu.arcs = .Call(call_tree_directions,
                        arcs = chow.liu.arcs,
                        nodes = explanatory,
                        root = root,
                        debug = FALSE)

  # merge learned and predetermined arcs.
  res$arcs = arcs.rbind(class.arcs, chow.liu.arcs)
  # update the network structure.
  res$nodes = cache.structure(nodes, arcs = res$arcs)
  # set a second class "bn.tan" to reroute method dispatch as needed.
  class(res) = c("bn.tan", "bn")

  return(res)

}#TAN.BACKEND





######################
## chow-liu algorithm
######################

chow.liu.backend = function(x, nodes, estimator, whitelist, blacklist,
    conditional = NULL, debug = FALSE) {

  # fix the whitelist and the blacklist to keep the C side simple.
  if (!is.null(blacklist)) {

    # arcs must be blacklisted in both directions, so keep only
    # the undirected ones.
    blacklist = blacklist[which.undirected(blacklist, nodes), , drop = TRUE]
    # keep only one direction for each blacklisted arc.
    blacklist = pdag2dag.backend(blacklist, nodes)

  }#THEN

  if (!is.null(whitelist)) {

    # keep only one direction for each whitelisted arc.
    whitelist = pdag2dag.backend(whitelist, nodes)

    # the chow-liu algorithms allows the selection of exactly length(nodes) arcs,
    # so the whitelist must contain less.
    if (nrow(whitelist) > length(nodes))
      stop("too many whitelisted arcs, there can be only ", length(nodes), ".")

  }#THEN

  .Call(call_chow_liu,
        data = x,
        nodes = nodes,
        estimator = estimator,
        whitelist = whitelist,
        blacklist = blacklist,
        conditional = conditional,
        debug = debug)

}#CHOW.LIU.BACKEND






################
## GRAPHS ## 
################
# create an empty graph from a given set of nodes.
empty.graph = function(nodes, num = 1) {

  random.graph(nodes = nodes, num = num, method = "empty", debug = FALSE)

}#EMPTY.GRAPH


# generate a random graph.
random.graph = function(nodes, num = 1, method = "ordered", ..., debug = FALSE) {

  # check the generation method.
  check.label(method, choices = graph.generation.algorithms,
    labels = graph.generation.labels, argname = "graph generation method",
    see = "random.graph")
  # check the node labels.
  check.nodes(nodes)
  # check the number of graph to generate.
  if (!is.positive.integer(num))
    stop(" the number of graphs to generate must be a positive integer number.")

  # expand and sanitize method-specific arguments.
  extra.args = check.graph.generation.args(method = method,
                 nodes = nodes, extra.args = list(...))

  random.graph.backend(num = num, nodes = nodes, method = method,
    extra.args = extra.args, debug = debug)

}#RANDOM.GRAPH




random.graph.backend = function(num, nodes, method, extra.args, debug = FALSE) {

  if (method == "ordered") {

    res = ordered.graph(num = num, nodes = nodes, prob = extra.args$prob)

  }#THEN
  else if (method == "ic-dag") {

    # adjust the number of graph to generate with the stepping factor.
    num = num * extra.args$every

    res = ide.cozman.graph(num = num, nodes = nodes,
            burn.in = extra.args$burn.in,
            max.in.degree = extra.args$max.in.degree,
            max.out.degree = extra.args$max.out.degree,
            max.degree = extra.args$max.degree,
            connected = TRUE, debug = debug)

    # keep only every k-th network.
    if (num > 1) {

      res = res[seq(from = extra.args$every, to = num, by = extra.args$every)]

    }#THEN

  }#THEN
  else if (method == "melancon") {

    # adjust the number of graph to generate with the stepping factor.
    num = num * extra.args$every

    res = ide.cozman.graph(num = num, nodes = nodes,
            burn.in = extra.args$burn.in,
            max.in.degree = extra.args$max.in.degree,
            max.out.degree = extra.args$max.out.degree,
            max.degree = extra.args$max.degree,
            connected = FALSE, debug = debug)

    # keep only every k-th network.
    if (num > 1) {

      res = res[seq(from = extra.args$every, to = num, by = extra.args$every)]

    }#THEN

  }#THEN
  else if (method == "empty") {

    res = empty.graph.backend(num = num, nodes = nodes)

  }#THEN

  return(res)

}#RANDOM.GRAPH.BACKEND



# generate a random directed acyclic graph.
ordered.graph = function(num, nodes, prob) {

  .Call(call_ordered_graph, ## need to find this in the C code files
        nodes = nodes,
        num = as.integer(num),
        prob = prob)

}#ORDERED.GRAPH



# generate a random directed acyclic graph accordin to a uniform
# probability distribution over the space of connected graphs (if
# connected = TRUE) or the space of graphs (if connected = FALSE).
ide.cozman.graph = function(num, nodes, burn.in, max.in.degree,
    max.out.degree, max.degree, connected, debug = FALSE) {

  .Call(call_ide_cozman_graph,
        nodes = nodes,
        num = as.integer(num),
        burn.in = as.integer(burn.in),
        max.in.degree = as.numeric(max.in.degree),
        max.out.degree = as.numeric(max.out.degree),
        max.degree = as.numeric(max.degree),
        connected = connected,
        debug = debug)

}#IDE.COZMAN.GRAPH




########################
## CHECKS ##
########################

# sanitize the extra arguments passed to Bayesian classifiers.
check.classifier.args = function(method, data, training, explanatory,
    extra.args) {

  if (method == "tree.bayes") {

    # check the label of the mutual information estimator.
    extra.args$estimator = check.mi.estimator(extra.args$estimator, data)

    # check the node to use the root of the tree (if not specified pick the first
    # explanatory variable assuming natural ordering).
    if (!is.null(extra.args$root))
      check.nodes(extra.args$root, graph = explanatory, max.nodes = 1)
    else
      extra.args$root = explanatory[1]

  }#THEN

  return(extra.args)

}#CHECK.CLASSIFIER.ARGS



# are all numeric values in the data frame fimite?
check.data.frame.finite = function(x) {

  .Call(call_data_frame_finite,
        data = x)

}#CHECK.DATA.FRAME.FINITE

# check logical flags.
check.logical = function(bool) {

  if (!is.logical(bool) || is.na(bool) || (length(bool) != 1))
    stop(sprintf("%s must be a logical value (TRUE/FALSE).",
           deparse(substitute(bool))))

}#CHECK.LOGICAL





# check labels for various arguments.
check.label = function(arg, choices, labels, argname, see) {

  if (!is.string(arg))
    stop("the ", argname, " must be a single character string.")

  if (arg %in% choices)
    return(invisible(NULL))

  # concatenate valid values, optinally with labels.
  if (missing(labels)) {

    choices = paste(paste('"', choices, '"', sep = ""), collapse = ", ")

  }#THEN
  else {

    labels = paste("(", labels[choices], ")", sep = "")
    choices = paste('"', choices, '"', sep = "")
    nl = length(labels)
    choices = paste(choices, labels, collapse = ", ")

  }#THEN

  # mention the most relevant manual page.
  if (missing(see))
    see = character(0)
  else
    see = paste(" See ?", see, " for details.", sep = "")

  # build the error message.
  errmsg = paste("valid ", argname, "(s) are ", choices, ".", see, sep = "")

  # print make sure that it is not truncated if possible at all.
  errlen = unlist(options("warning.length"), use.names = FALSE)
  options("warning.length" = max(1000, min(8170, nchar(errmsg) + 20)))

  stop(errmsg)

  options("warning.length" = errlen)

}#CHECK.LABEL




# generate an empty 'bn' object given a set of nodes.
empty.graph.backend = function(nodes, num = 1) {

  .Call(call_empty_graph,
        nodes = nodes,
        num = as.integer(num))

}#EMPTY.GRAPH.BACKEND






