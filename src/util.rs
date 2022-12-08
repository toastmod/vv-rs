pub fn do_while(_do: &mut dyn FnMut(), _while: &mut dyn FnMut() -> bool) {
    loop {
        _do();
        if _while() {
            break;
        }
    }
}