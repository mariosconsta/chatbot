from tkinter import * #graphical user interphase. to * tha kanei import ta panta
from chat import response, bot_name #apo to chat theloyme to response function kai to onoma toy bot

BG_GRAY = "#ABB2B9" #xrwmata gia to peribalon
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14" #grammatoseira kai megethos
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    
    def __init__(self):
        self.window = Tk() #anoigei to peribalon se parathyro
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("Chat") #titlos toy peribalontos
        self.window.resizable(width=False, height=False) #den allazei to megethos toy parathyroy
        self.window.configure(width=470, height=550, bg=BG_COLOR) #platos, mhkos, xrwma
        
        # head label #epikefalida,xrwma epikefalidas kai xrwma grammatwn
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10) #ti grafei h epikefalida kai poy brisketai
        head_label.place(relwidth=1)
        
        # tiny divider #diaxwrismos ths epikefalidas apo to ypoloipo parathyro
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # text widget #edw tha brisketai h synomilia toy xrhsth me to bot, dhladh ayta poy exei grapsei o xrhsths kai oi apanthseis toy bot
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, #tha exei 20 xarakthres ana grammh kai tha xrhsimopoiei 2 grammes
                                font=FONT, padx=5, pady=5) #poy tha brisketai sto parathyro
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)#xrhsimopoioyme peripoy to 75% gia ayto to meros toy periballontos
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        
        # scroll bar 
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974) # poy tha brisketai sto parathyro
        scrollbar.configure(command=self.text_widget.yview)
        
        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        
        # message entry box # edw grafei o xrhsths
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT) #xrwma periballontos kai grammatwn
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011) #poy brisketai sto peribalon
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY, #ti grafei to koympi "send", megethos kai ti xrwma tha exei
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)# poy brisketai
     
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get() #pairnei to message san string
        self._insert_message(msg, "You")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return # se periptwsh poy den exei grapsei o xrhsths kati
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n" #dhmioyrgoyme to mhnyma kai afhnei dyo kenes seires
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {response(msg)}\n\n" #apanthsh kai 2 kenes seires
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END) # tha mporoyme panta na doyme to teleytaio mhnyma
             
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()