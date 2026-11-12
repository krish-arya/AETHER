"""Configuration of the technical feature library (PRD 6.2).

Every window is configurable; changing any value changes ``feature_set_version`` so
feature sets computed under different settings can never be confused.
"""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

# Bump when indicator maths or feature naming changes without a config change.
FEATURE_CODE_VERSION = "1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MacdConfig(_Strict):
    fast: int = Field(12, ge=2)
    slow: int = Field(26, ge=3)
    signal: int = Field(9, ge=2)


class StochasticConfig(_Strict):
    k: int = Field(14, ge=2)
    d: int = Field(3, ge=1)


class BollingerConfig(_Strict):
    window: int = Field(20, ge=2)
    num_std: float = Field(2.0, gt=0)


class FeatureConfig(_Strict):
    sma_windows: list[int] = [20, 50]
    ema_windows: list[int] = [12, 26]
    macd: MacdConfig = MacdConfig()
    adx_window: int = Field(14, ge=2)
    rsi_window: int = Field(14, ge=2)
    stochastic: StochasticConfig = StochasticConfig()
    roc_window: int = Field(10, ge=1)
    williams_window: int = Field(14, ge=2)
    atr_window: int = Field(14, ge=2)
    bollinger: BollingerConfig = BollingerConfig()
    hist_vol_windows: list[int] = [20, 60]
    rolling_std_window: int = Field(20, ge=2)
    obv_window: int = Field(20, ge=2)
    vwap_window: int = Field(20, ge=1)
    mfi_window: int = Field(14, ge=2)
    cum_return_windows: list[int] = [5, 21, 63]
    beta_window: int = Field(60, ge=5)
    corr_window: int = Field(60, ge=5)
    trading_days_per_year: int = Field(252, ge=1)

    def version(self) -> str:
        payload = json.dumps(
            {"code": FEATURE_CODE_VERSION, "config": self.model_dump(mode="json")},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
