# Titanic survival prediction with logistic regression (R port)
#
# Usage: Rscript titanic_logreg.R [-ht | --hide_test_preds]
# Requires: install.packages("glmnet")

suppressPackageStartupMessages(library(glmnet))

args <- commandArgs(trailingOnly = TRUE)
hide_test_preds <- any(args %in% c("-ht", "--hide_test_preds"))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

pop_sd <- function(x) sqrt(mean((x - mean(x))^2))


fit_preprocessor <- function(df) {
  age_mean  <- mean(df$Age, na.rm = TRUE)
  fare_mean <- mean(df$Fare, na.rm = TRUE)
  age_filled  <- ifelse(is.na(df$Age), age_mean, df$Age)
  fare_filled <- ifelse(is.na(df$Fare), fare_mean, df$Fare)

  list(
    embarked_levels = sort(unique(df$Embarked[!is.na(df$Embarked)])),
    pclass_levels   = sort(unique(df$Pclass)),
    age_mean        = age_mean,
    fare_mean       = fare_mean,
    age_center      = mean(age_filled),
    age_scale       = pop_sd(age_filled),
    fare_center     = mean(fare_filled),
    fare_scale      = pop_sd(fare_filled)
  )
}


transform_features <- function(df, prep) {
  age  <- ifelse(is.na(df$Age), prep$age_mean, df$Age)
  fare <- ifelse(is.na(df$Fare), prep$fare_mean, df$Fare)

  X <- data.frame(
    Age   = (age - prep$age_center) / prep$age_scale,
    Fare  = (fare - prep$fare_center) / prep$fare_scale,
    Sex   = ifelse(tolower(df$Sex) == "male", 1, 0),
    SibSp = df$SibSp,
    Parch = df$Parch
  )

  for (lv in prep$embarked_levels) {
    X[[paste0("Embarked_", lv)]] <- as.numeric(!is.na(df$Embarked) & df$Embarked == lv)
  }
  for (lv in prep$pclass_levels) {
    X[[paste0("Pclass_", lv)]] <- as.numeric(df$Pclass == lv)
  }

  rownames(X) <- df$PassengerId
  X
}

prep_data <- function(raw_df, prep = NULL, verbose = FALSE) {
  use_cols <- c("PassengerId", "Sex", "Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked")
  if ("Survived" %in% names(raw_df)) use_cols <- c(use_cols, "Survived")
  df <- raw_df[, use_cols]

  if (verbose) {
    cat("Columns used for model construction: 'Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'.\n")
  }

  if (is.null(prep)) prep <- fit_preprocessor(df)

  if (verbose) {
    cat("One-hot encoding 'Embarked' feature...\n")
    cat("One-hot encoded 'Embarked' columns:", paste0("Embarked_", prep$embarked_levels), "\n")
    cat("One-hot encoding 'Pclass' ('Passenger class') feature...\n")
    cat("One-hot encoded 'Pclass' feature names:", paste0("Pclass_", prep$pclass_levels), "\n")
    cat(sprintf("Found %d null values in 'Age' column. Imputing empty values...\n", sum(is.na(df$Age))))
    cat("Scaling 'Age' and 'Fare' features...\n")
  }

  X <- transform_features(df, prep)
  y <- if ("Survived" %in% names(df)) df$Survived else NULL

  list(X = X, y = y, prep = prep)
}


# ---------------------------------------------------------------------------
# Modeling helpers
# ---------------------------------------------------------------------------

# Stratified train/test split; returns the row indices of the test set
stratified_test_idx <- function(y, test_size, seed) {
  set.seed(seed)
  idx_by_class <- split(seq_along(y), y)
  test_idx <- unlist(lapply(idx_by_class, function(idx) {
    n_test <- round(length(idx) * test_size)
    idx[sample.int(length(idx), n_test)]
  }))
  sort(unname(test_idx))
}

safe_div <- function(a, b) if (b == 0) 0 else a / b

classification_scores <- function(y_true, y_pred) {
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  precision <- safe_div(tp, tp + fp)
  recall    <- safe_div(tp, tp + fn)
  list(
    accuracy  = mean(y_true == y_pred),
    f1        = safe_div(2 * precision * recall, precision + recall),
    precision = precision,
    recall    = recall
  )
}

predict_class <- function(model, X, lambda) {
  probs <- predict(model, newx = as.matrix(X), s = lambda, type = "response")
  as.integer(as.numeric(probs) > 0.5)
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if (sys.nframe() == 0) {
  raw_train_df <- read.csv("../data/train.csv", na.strings = c("", "NA"), stringsAsFactors = FALSE)
  test_df      <- read.csv("../data/test.csv",  na.strings = c("", "NA"), stringsAsFactors = FALSE)

  prepped <- prep_data(raw_train_df, verbose = TRUE)
  X <- prepped$X
  y <- prepped$y

  test_idx <- stratified_test_idx(y, test_size = 0.15, seed = 93)
  X_train <- X[-test_idx, ]; y_train <- y[-test_idx]
  X_test  <- X[test_idx, ];  y_test  <- y[test_idx]

  cat("\nTraining model...\n")
  # sklearn's LogisticRegression defaults to L2 regularization with C = 1.
  # The equivalent glmnet setting is a ridge penalty (alpha = 0) with
  # lambda = 1 / (C * n) and no internal standardization.
  lambda <- 1 / nrow(X_train)
  model <- glmnet(as.matrix(X_train), y_train, family = "binomial",
                  alpha = 0, lambda = lambda, standardize = FALSE)

  train_scores <- classification_scores(y_train, predict_class(model, X_train, lambda))
  cat("\n---TRAIN-TEST SCORES---\n")
  cat("Model training accuracy score:",  round(train_scores$accuracy, 4),  "\n")
  cat("Model training f1 score:",        round(train_scores$f1, 4),        "\n")
  cat("Model training precision score:", round(train_scores$precision, 4), "\n")
  cat("Model training recall score:",    round(train_scores$recall, 4),    "\n")

  test_scores <- classification_scores(y_test, predict_class(model, X_test, lambda))
  cat("\nModel test accuracy score:",  round(test_scores$accuracy, 4),  "\n")
  cat("Model test f1 score:",        round(test_scores$f1, 4),        "\n")
  cat("Model test precision score:", round(test_scores$precision, 4), "\n")
  cat("Model test recall score:",    round(test_scores$recall, 4),    "\n")

  # Transform the competition test set using parameters learned on the training data
  test_data <- prep_data(test_df, prep = prepped$prep)$X
  final_test_preds <- data.frame(
    PassengerId = test_df$PassengerId,
    Survived    = predict_class(model, test_data, lambda)
  )

  if (!hide_test_preds) {
    cat("\n---TEST PREDICTIONS---\n")
    print(final_test_preds, row.names = FALSE)
    cat("(Pass '-ht' / '--hide_test_preds' command line argument to only show train-test scores)\n")
  } else {
    cat("\nFinal test predictions hidden.\n")
  }
}