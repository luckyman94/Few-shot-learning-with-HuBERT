import torch


def compute_prototypes(
    model,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    device=None,
):
    

    model.eval()

    if device is not None:
        X_support = X_support.to(device)
        y_support = y_support.to(device)

    classes = torch.unique(y_support)

    prototypes = []
    proto_labels = []

    with torch.no_grad():
        for c in classes:
            Xc = X_support[y_support == c]

            out = model(Xc)
            h = out.last_hidden_state          
            z = h.mean(dim=1)                  

            proto = z.mean(dim=0)              
            prototypes.append(proto)
            proto_labels.append(c)

    return torch.stack(prototypes), torch.stack(proto_labels)


def classify(
    model,
    X_query: torch.Tensor,
    prototypes: torch.Tensor,
    proto_labels: torch.Tensor,
    device=None,
):


    model.eval()

    if device is not None:
        X_query = X_query.to(device)
        prototypes = prototypes.to(device)
        proto_labels = proto_labels.to(device)

    with torch.no_grad():
        out = model(X_query)
        h = out.last_hidden_state             
        z = h.mean(dim=1)                     

        dists = torch.cdist(z, prototypes)    
        pred_idx = dists.argmin(dim=1)
        preds = proto_labels[pred_idx]

    return preds



