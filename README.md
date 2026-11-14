# GPz 2.0

**Sparse Gaussian process regression with input-dependent noise, uncertain inputs, and missing features.**

GPz learns a regression model using a set of Gaussian basis functions. It returns a prediction and an estimate of its variance, with separate contributions from uncertainty in the fitted model, output noise, and uncertainty in the inputs. The basis centres and scales are learned during training.

Developed by **Ibrahim Almosallam**. The model and its derivations are described in the [doctoral thesis](https://ora.ox.ac.uk/objects/uuid%3A6a3b600d-5759-456a-b785-5f89cf4ede6d).

## Requirements

This version has been tested with **MATLAB R2026b** and requires:

- **Optimization Toolbox** for `fminunc`.
- **Statistics and Machine Learning Toolbox** for the statistical routines used by the implementation and demos.

## Getting started

Set MATLAB's Current Folder to the directory containing this README and the demo scripts, then add the implementation to the path:

```matlab
addpath('GPz');
```

Run a demo by entering its name in the Command Window:

| Script | Example |
| --- | --- |
| `demo_sinc` | One-dimensional regression with varying output noise and a gap in the observations. |
| `demo_2D` | Two-dimensional regression, including predictions when an input feature is missing. |
| `demo_photoz` | Photometric redshift estimation using the included `data/sdss_sample.csv` dataset. |

Each demo has a configuration section near the top. Adjust `m`, `method`, `maxIter`, and the noise settings there. Larger values of `m` increase the cost of training and prediction.

## A small example

The following example generates data, fits a model, and evaluates predictions on a held-out test set. Run it from the same directory as the demos.

```matlab
addpath('GPz');
rng(1);

X = linspace(-3,3,240)';
Y = sin(X) + (0.10 + 0.05*(X+3)).*randn(size(X));

[training,validation,testing] = sample(size(X,1),0.7,0.15,0.15);

method = 'VL';
m = 10;

% Initialize on the training rows so normalization uses the training data.
model = init(X(training,:),Y(training,:),method,m, ...
    'normalize',true,'heteroscedastic',true);

model = train(model,X,Y, ...
    'training',training,'validation',validation,'testing',testing, ...
    'maxIter',100,'maxAttempts',10);

[mu,sigma,nu,beta_i,gamma] = predict(X,model,'selection',testing);
rmse = sqrt(mean((Y(testing)-mu).^2));
fprintf('Test RMSE: %.4f\n',rmse);
```

`sample` makes a random split and returns logical column vectors. For a chronological or extrapolation experiment, construct `training`, `validation`, and `testing` yourself, with validation and test observations following the training interval. Use disjoint sets; `testing` is used only when evaluating the fitted model.

## Data and predictions

The MATLAB interface stores **one observation per row**:

| Variable | Size | Meaning |
| --- | --- | --- |
| `X` | `n` by `d` | Input features. Use `NaN` for missing entries. |
| `Y` | `n` by `k` | Observed targets; use a column vector for one target. |
| `training`, `validation`, `testing` | `n` by `1` logical | Rows assigned to each subset. |
| `omega` | `n` by `1` | Optional sample weights. |

Keep the same feature order and number of columns at prediction time. Mark an unavailable feature with `NaN`; keep its column in `X`. Targets used for fitting and validation must be observed.

The main prediction outputs are:

```matlab
[mu,sigma,nu,beta_i,gamma] = predict(X,model);
```

| Output | Meaning |
| --- | --- |
| `mu` | Predicted target mean. |
| `sigma` | Total predictive variance: `nu + beta_i + gamma`. |
| `nu` | Variance from uncertainty in the regression weights. |
| `beta_i` | Output-noise contribution to the predictive variance. |
| `gamma` | Contribution from variation of the predicted mean over uncertain or missing inputs. |

Each of these outputs has one row per selected observation and one column per target. **`sigma` is a variance, not a standard deviation.** For a two-standard-deviation band, use `mu - 2*sqrt(sigma)` and `mu + 2*sqrt(sigma)`.

Predictions use `model.best` by default. To use the final optimizer iterate instead:

```matlab
[mu,sigma] = predict(X,model,'whichSet','last');
```

## Choosing the basis functions

Set `method` when calling `init`. The six choices control how the Gaussian basis scales are shared:

| Method | Covariance structure |
| --- | --- |
| `GL` | One isotropic scale shared by all basis functions. |
| `VL` | A separate isotropic scale for each basis function. |
| `GD` | One diagonal covariance shared by all basis functions. |
| `VD` | A separate diagonal covariance for each basis function. |
| `GC` | One full covariance shared by all basis functions. |
| `VC` | A separate full covariance for each basis function. |

Diagonal covariances allow a different scale along each input coordinate. Full covariances also allow rotated basis functions. All methods learn the basis centres. With a single input feature, the implementation uses `GL` for global methods and `VL` for variable methods.

## Initialization and preprocessing

```matlab
model = init(X,Y,method,m, ...
    'normalize',true,'heteroscedastic',true);
```

| Option | Default | Effect |
| --- | --- | --- |
| `normalize` | `true` | Subtract each input feature's mean and divide by its standard deviation, ignoring missing entries. |
| `heteroscedastic` | `true` | Learn output noise that varies with the input. Set to `false` for a constant output-noise variance per target. |
| `training` | All rows | Select the rows used for normalization and model initialization. |
| `Psi` | `[]` | Supply input measurement variances or covariances. |

Normalization statistics are computed only from the rows selected by `training`. If that option is omitted, all rows supplied to `init` are used. The small example above supplies only training rows during initialization. Keep nonconstant observed features when using normalization.

The model stores `muX`, `sdX`, and the training-target mean `muY`. `train` and `predict` apply the stored input transformation internally. Targets are centred during fitting and their mean is restored in predictions; target standard deviations are not used to rescale them. Supply inputs and input uncertainties in their original units to these functions.

## Input measurement uncertainty

Pass input measurement uncertainty through the `Psi` option. Use **variances or covariances**, rather than measurement standard deviations.

For a consistent representation across initialization, training, and prediction:

| Method | Shape of `Psi` |
| --- | --- |
| `GL`, `VL`, `GD`, `VD` | `n` by `d`, containing a variance for each input entry. |
| `GC`, `VC` | `d` by `d` by `n`, with one input covariance matrix per observation. |

For example, if `E` contains measurement standard deviations and `method` is `VD`:

```matlab
Psi = E.^2;

model = init(X(training,:),Y(training,:),method,m, ...
    'Psi',Psi(training,:));
model = train(model,X,Y, ...
    'training',training,'validation',validation,'testing',testing,'Psi',Psi);

[mu,sigma] = predict(X,model,'Psi',Psi,'selection',testing);
```

For `GC` or `VC`, select covariance pages with `Psi(:,:,training)` when initializing on training rows. Supply symmetric positive semidefinite covariance matrices in the original feature units.

At prediction time, `Psi` describes the uncertainty of the inputs being predicted. When using `selection`, provide `Psi` for all rows in the accompanying `X`; `predict` selects the corresponding entries internally. Use `Psi = []` to omit measurement uncertainty. Missing features are represented separately by `NaN` in `X`.

## Training controls and progress

| Option to `train` | Default | Meaning |
| --- | --- | --- |
| `maxIter` | `200` | Maximum number of optimizer iterations. |
| `maxAttempts` | `Inf` | Number of consecutive non-improving validation iterations allowed before stopping. |
| `training` | All rows | Observations used to fit the parameters. |
| `validation` | `[]` | Observations used for model selection and early stopping. |
| `testing` | `[]` | Held-out observations scored only for the progress plot. |
| `omega` | Ones | Sample weights. |
| `Psi` | `[]` | Input measurement uncertainty. |

Training opens a live progress figure with three panels: the training objective, RMSE, and mean log likelihood (MLL). The RMSE and MLL panels show training, validation, and testing curves when the corresponding subsets are supplied. Axis limits adjust automatically as points are added. The title reports the iteration, elapsed time, and stopping status. Closing the progress window leaves training running.

The three panels are stacked vertically, with a shared colour legend at the top and current metric values in legends on the right. Blue denotes training, red validation, and green testing. A black dashed line on each panel marks the iteration selected by the best validation MLL; the top legend gives its score and iteration. These markers move when a new best is selected, including ties. When resuming from a model whose saved best precedes the displayed run, the legend labels it as belonging to the previous run, and no iteration is marked until the current run matches or improves that score.

The `logML/n` panel displays the negative of the objective minimized by the optimizer; that objective is normalized by the number of training rows and targets. The RMSE and MLL curves describe the current accepted iterate, rather than the best model found so far.

All three MLL curves use the fitted mean and output-noise variance only. A separately calculated MLL using the `sigma` returned by `predict` includes all three predictive variance contributions and can therefore differ from the plotted metric.

Use disjoint logical masks for `training`, `validation`, and `testing`. Test scores are evaluated separately for display at accepted iterates. They do not enter the optimization objective, gradients, validation stopping counter, or best-model selection. Omitting `testing` avoids the additional evaluation. Input normalization uses training rows, and the photometric-redshift demo derives sample weights separately within each subset.

With validation enabled, a validation MLL that matches or improves on the best score resets the stopping counter. `model.best` retains that selected model, while `model.last` stores the final iterate. Without validation, both refer to the final iterate and `maxAttempts` is inactive, even if a testing curve is shown.

To continue a fit, pass the returned `model` back to `train` with the same data, weights, and subset masks. Optimization resumes from `model.last`; the saved validation best is retained for comparison.

## Sample weights

For a scalar target, `getOmega` provides three weighting schemes:

| Call | Weights |
| --- | --- |
| `getOmega(Y,'normal')` | Equal weights. |
| `getOmega(Y,'balanced',binWidth)` | Inverse target-bin counts, scaled by the largest bin count. |
| `getOmega(Y,'normalized')` | `(1 + Y).^(-2)`, as used for redshift-normalized squared errors. |

Supply the resulting vector to `train` through `'omega',omega`; each subset uses its corresponding rows. For data-dependent weighting such as `'balanced'`, derive training weights using only training targets. The photometric-redshift demo computes weights independently within each subset, so changing the test targets cannot affect training or validation weights. Weights change the fitting objective and progress metrics; they are separate from input normalization and from `Psi`.

## Reference

Ibrahim Almosallam (2017). *Heteroscedastic Gaussian Processes for Uncertain and Incomplete Data*. Doctoral thesis, University of Oxford. [Oxford Research Archive](https://ora.ox.ac.uk/objects/uuid%3A6a3b600d-5759-456a-b785-5f89cf4ede6d).
