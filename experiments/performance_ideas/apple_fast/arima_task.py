"""Actual AutoARIMA task-quality fixture, independent of private Kalman probes."""
from support import consumed,binding_check

def search_cases(args):
    import numpy as np
    from mojolearn import AutoARIMA,ARIMA
    binding=ARIMA()._extension()
    cases={}
    for length in (129,257,509):
        rng=np.random.default_rng(883);horizon=17
        innovations=rng.normal(size=(3,length+horizon)).astype('float32')
        series=np.zeros_like(innovations)
        for t in range(1,length+horizon):  # test-only independent DGP, never runtime
            series[0,t]=.7*series[0,t-1]+innovations[0,t]
            series[1,t]=.97*series[1,t-1]+innovations[1,t]
            series[2,t]=series[2,t-1]+.08+innovations[2,t]
        train=np.ascontiguousarray(series[:,:length])
        model=AutoARIMA(train)
        def fit():
            model.search(s=1,d=range(2),p=range(3),q=range(2),P=range(1),D=range(1),Q=range(1),maxiter=20)
            model.fit(maxiter=100)
            return model.forecast(horizon)
        forecast,elapsed=consumed(fit)
        forecast=np.asarray(forecast,float)
        error=float(np.sqrt(np.mean((forecast-series[:,length:])**2)))
        fit_status=[]
        for fitted in model._fitted:
            assert np.isfinite(np.asarray(fitted.params_)).all()
            fit_status.append(dict(iterations=np.asarray(fitted.n_iter_).tolist(),retcode=np.asarray(fitted.retcode_).tolist(),loglike=np.asarray(fitted.llf_).tolist()))
        cases[str(length)]=dict(contract=dict(series=3,length=length,horizon=horizon,seed=883,orders=[3,2,2],maxiter=100),
            metrics=dict(forecast_rmse=dict(value=error,rtol=.001,atol=1e-5)),fit_search_forecast_ms=elapsed,
            orders=np.asarray(model.order_).tolist(),criteria=np.asarray(model.ic_).tolist(),optimizer=fit_status)
    return dict(binding=binding_check(binding,'arima'),cases=cases)
