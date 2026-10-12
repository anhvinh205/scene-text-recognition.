import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from scenetext.data.vocab import Vocab  # noqa: E402
from scenetext.models.crnn import CRNN  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    return CRNN(
        Vocab().num_classes, backbone="resnet18", hidden_size=32, n_layers=2, pretrained=False
    )


def test_output_shape_and_log_softmax(model):
    model.eval()
    out = model(torch.randn(2, 1, 100, 420))
    t, b, c = out.shape
    assert (b, c) == (2, 37)
    assert t == 53, "stride relaxation should keep W/8 time steps"
    torch.testing.assert_close(out.exp().sum(-1), torch.ones(t, b), atol=1e-4, rtol=1e-4)


def test_ctc_loss_backward_and_overfits_one_batch():
    torch.manual_seed(0)
    v = Vocab()
    m = CRNN(v.num_classes, backbone="resnet18", hidden_size=32, n_layers=1, pretrained=False)
    crit = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    x = torch.randn(4, 1, 100, 420)
    texts = ["abc", "hello", "z9", "text"]
    ml = max(map(len, texts))
    tgt = torch.zeros(4, ml, dtype=torch.long)
    for i, s in enumerate(texts):
        tgt[i, : len(s)] = torch.tensor(v.encode(s))
    lens = torch.tensor([len(s) for s in texts])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = last = None
    for step in range(25):
        lp = m(x)
        loss = crit(lp, tgt, torch.full((4,), lp.shape[0], dtype=torch.long), lens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    assert last < first


def test_parameter_groups_respect_freezing():
    m = CRNN(
        37, backbone="resnet18", hidden_size=16, n_layers=1, pretrained=False, freeze_backbone=True
    )
    assert len(m.parameter_groups(1e-3)) == 1
