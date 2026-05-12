import joblib
import torch
import numpy as np

from srfn_model import StatisticalResidualFusionNet

torch.manual_seed(42)
np.random.seed(42)

def load_models():

    lgbm = joblib.load("models/lightgbm.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    model = StatisticalResidualFusionNet(
        num_classes=len(label_encoder.classes_),
        num_expert_features=389
    )

    model.load_state_dict(torch.load("models/srfn_model.pth", map_location="cpu"))
    model.eval()

    return lgbm, model, scaler, label_encoder


def predict_modulation(reduced_df, raw_df, snr, lgbm, srfn, scaler, label_encoder):

    # ---------------- LIGHTGBM ----------------
    if snr > 2:
        X = reduced_df.drop(columns=["modulation", "snr"], errors='ignore')
        pred = lgbm.predict(X)[0]

        # ✅ FIX: Convert numeric label → modulation name
        modulation = label_encoder.inverse_transform([pred])[0]

        return modulation, "LightGBM", "High (~90%)"

    # ---------------- SRFN ----------------
    else:
        try:
            X_exp = reduced_df.drop(columns=["modulation", "snr"], errors='ignore').values
            X_exp = scaler.transform(X_exp)

            sig_cols = [col for col in raw_df.columns if col.startswith("sig_")]
            raw = raw_df[sig_cols].values.astype(np.float32)

            I = raw[:, :128]
            Q = raw[:, 128:]

            iq = np.stack([I, Q], axis=1)

            iq_tensor = torch.tensor(iq, dtype=torch.float32)
            exp_tensor = torch.tensor(X_exp, dtype=torch.float32)

            with torch.no_grad():
                output = srfn(iq_tensor, exp_tensor)
                pred = torch.argmax(output, dim=1).item()

            # Already correct decoding
            modulation = label_encoder.inverse_transform([pred])[0]

            return modulation, "SRFN", "Medium (~70%)"

        except Exception as e:
            print("SRFN ERROR:", e)
            return "SRFN Error", "SRFN", "0%"